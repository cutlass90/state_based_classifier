from __future__ import division, print_function

import os
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import classifier_tools as utils
from parameters import *
import misc

print('model 28$$$$')

class Classifier:

    def __init__(self, batch_size, n_beats = 10, overlap = 5,
        n_channel = 3, nHiddenRNN = 64,
        nHiddenFC = 64, dropout = 1, do_train = True):

        self.batch_size = batch_size
        self.n_beats = n_beats
        self.overlap = overlap
        self.n_channel = n_channel
        self.nHiddenRNN = nHiddenRNN
        self.nHiddenFC = nHiddenFC
        self.dropout = dropout
        self.l2Koeff = 0.01
        self.create_graph()
        if do_train: self.create_optimizer_graph(self.cost)
        self.train_writer = tf.train.SummaryWriter(logdir = 'summary/')
        self.merged = tf.merge_all_summaries()
        
        init_op = tf.initialize_all_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(init_op)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(),
                                    max_to_keep = 1000)

    # --------------------------------------------------------------------------
    def __enter__(self):
        return self

    # --------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.reset_default_graph()
        if self.sess is not None:
            self.sess.close()

    # --------------------------------------------------------------------------
    def create_RNN_graph(self, inputs, sequence_length):
        with tf.variable_scope('RNN_graph'):
            # inputs b*(n_b+o) x h x c (h is variable value)
            # sequence_length b*(n_b+o)
            fw_cell = tf.nn.rnn_cell.GRUCell(self.nHiddenRNN)
            bw_cell = tf.nn.rnn_cell.GRUCell(self.nHiddenRNN)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32)
        return states # tuple of fw and bw states with shape b*(n_b+o) x hRNN

    # --------------------------------------------------------------------------
    def create_state_RNN_graph(self, inputs):
        # inputs b x (n_b+o) x 2*hRNN
        with tf.variable_scope('state_RNN_graph'):
            # inputs preprocessing
            fw_inputs = inputs[:,:self.n_beats,:] #b x n_b x 2*hRNN
            bw_inputs = tf.reverse(inputs, [False, True, False]) #b x (n_b+o) x 2*hRNN

            fw_cell = tf.nn.rnn_cell.GRUCell(self.nHiddenRNN)
            bw_cell = tf.nn.rnn_cell.GRUCell(self.nHiddenRNN)
            states = tf.Variable(tf.zeros([self.batch_size, self.nHiddenRNN]),
                trainable=False)
            fw_RNN_outputs, new_state = tf.nn.dynamic_rnn(fw_cell, fw_inputs,
                initial_state = states, dtype=tf.float32, scope = 'fw_RNN') #b x n_b x hRNN
            with tf.control_dependencies([tf.assign(states, new_state)]):
                fw_RNN_outputs = tf.identity(fw_RNN_outputs)
            bw_RNN_outputs, _ = tf.nn.dynamic_rnn(bw_cell, bw_inputs,
                dtype=tf.float32, scope = 'bw_RNN') #b x (n_b+o) x hRNN
            bw_RNN_outputs = bw_RNN_outputs[:, self.overlap:, :] #b x n_b x hRNN
            bw_RNN_outputs = tf.reverse(bw_RNN_outputs, [False, True, False])

            fw_rs = tf.reshape(fw_RNN_outputs, shape=[self.batch_size*self.n_beats,
                self.nHiddenRNN]) #b*n_b x hRNN    b0s0, b0s1, b0s2...

            bw_rs = tf.reshape(bw_RNN_outputs, shape=[self.batch_size*self.n_beats,
                self.nHiddenRNN]) #b*n_b x hRNN    b0s0, b0s1, b0s2...
            RNNs = tf.concat(1, (fw_rs, bw_rs))#b*n_b x 2*hRNN

        return RNNs

    # --------------------------------------------------------------------------
    def create_FC_graph(self, inputs):
        with tf.variable_scope('FC_graph'):
            out_FC1 = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=self.nHiddenFC,
                activation_fn=tf.nn.elu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer,
                trainable=True) #b*(n_b+o) x hFC

            out_FC2 = tf.contrib.layers.fully_connected(
                inputs=out_FC1,
                num_outputs=self.nHiddenFC,
                activation_fn=tf.nn.elu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer,
                trainable=True) #b*(n_b+o) x hFC
        return out_FC2

    # --------------------------------------------------------------------------
    def create_cost_graph(self, logits, targets):
        #logits and targets # b*n_b x len(REQUIRED_DISEASES)

        self.sigmoid_cross_entropy = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
            logits=logits,
            targets=targets,
            pos_weight = 2,
            name='sigmoid_cross_entropy')) #b*(n_b+o) x len(REQUIRED_DISEASES)

        self.l2Loss = self.l2Koeff*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])

        tf.scalar_summary('Crossentropy', self.sigmoid_cross_entropy)
        tf.scalar_summary('L2 loss', self.l2Loss)
        
        return self.sigmoid_cross_entropy + self.l2Loss

    def create_optimizer_graph(self, cost):
        with tf.variable_scope('optimizer_graph'):
            optimizer = tf.train.AdamOptimizer(0.0001)

            #grad clipping
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 0.3)
            self.train = optimizer.apply_gradients(zip(grads, tvars))

    # --------------------------------------------------------------------------
    def create_graph(self):
        ####Define graph
        print('Creating graph ... ', end='')

        ####Graph input
        self.inputs = tf.placeholder(tf.float32,
            shape=[self.batch_size*(self.n_beats+self.overlap), None, self.n_channel],
            name='inputs') #b*(n_b+o) x h x c (h is variable value)
        self.target_events = tf.placeholder(tf.float32,
            shape=[self.batch_size*(self.n_beats+self.overlap), len(REQUIRED_DISEASES)],
            name='target_events') # b*(n_b+o) x len(REQUIRED_DISEASES)
        self.sequence_length = tf.placeholder(tf.int32,
            shape=[self.batch_size*(self.n_beats+self.overlap)])
        self.keep_prob = tf.placeholder(tf.float32)

        states = self.create_RNN_graph(self.inputs, self.sequence_length)

        states_con = tf.concat(1, states) #b*(n_b+o) x 2*hRNN

        states_rs = tf.reshape(states_con,
            [self.batch_size, self.n_beats+self.overlap, 2*self.nHiddenRNN])
            #b x (n_b+o) x 2*hRNN

        RNNs = self.create_state_RNN_graph(states_rs) #b*n_b x 2*hRNN

        FC = self.create_FC_graph(RNNs) #b*(n_b+o) x hFC

        logits = tf.contrib.layers.fully_connected(
            inputs=FC,
            num_outputs=len(REQUIRED_DISEASES),
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer,
            trainable=True) #b*(n_b+o) x len(REQUIRED_DISEASES)

        self.predicted_events = tf.sigmoid(logits) #b*(n_b+o) x len(REQUIRED_DISEASES)

        self.cost = self.create_cost_graph(logits,
            self.target_events[:self.batch_size*self.n_beats, :]) #scalar
        print('Done!')
        
    ############################################################################  
    def save_model(self, path = 'beat_detector_model', step = None):
        p = self.saver.save(self.sess, path, global_step = step)
        print("\nModel saved in file: %s" % p)

    ############################################################################
    def load_model(self, path):
        #path is path to file or path to directory
        #if path it is path to directory will be load latest model
        load_path = os.path.splitext(path)[0]\
        if os.path.isfile(path) else tf.train.latest_checkpoint(path)
        print('try to load {}'.format(load_path))
        self.saver.restore(self.sess, load_path)
        print("Model restored from file %s" % load_path)

    ############################################################################
    def train_(self, data_loader, n_iter = 10000, save_model_every_n_iter = 1000, path_to_model = 'classifier'):
        #if n_iter = None number of iterations are selected in way that one epoch would be train
        print('\n\n\n\t----==== Training ====----')
        #try to load model
        try:
            self.load_model(os.path.dirname(path_to_model))
        except:
            print('Can not load model {0}, starting new train'.format(path_to_model))
            
        start_time = time.time()
        
        for current_iter in tqdm(range(n_iter)):
            batch = data_loader.get_batch()
            feedDict = {self.inputs : batch['delta_coded_data'],
                        self.target_events : batch['events'],
                        self.sequence_length : batch['sequence_length'],
                        self.keep_prob : self.dropout}
            _, summary = self.sess.run([self.train, self.merged], feed_dict = feedDict)
            self.train_writer.add_summary(summary, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path = path_to_model, step = current_iter+1)

        self.save_model(path = path_to_model, step = current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))


    #############################################################################################################
    def predicting_events_with_chunks(self, path_to_file, n_chunks=128, overlap=700, path_to_predicted_beats = None, path_to_model = 'models/cl'):
        predicting_time = time.time()
        print('\n\n\n\t----==== Predicting beats ====----')
        #load model
        self.load_model(path_to_model)
        
        data = np.load(path_to_file).item()
        
        #padding end of file with zeros
        channels = misc.get_channels(data)
        ch_len = len(channels[0])
        padd_len = 2 * self.n_step * self.windows_size + self.overlap * self.windows_size
        channels = [np.concatenate((channel, np.linspace(channel[-1], 0, padd_len)), axis = 0) for channel in channels]
        data = misc.write_channels(data, channels)

        chunked_data = utils.chunking_data(data, overlap=overlap, n_chunks=n_chunks)
        generators = []
        for i in range(len(chunked_data)):
            gen = utils.step_generator(chunked_data[i],
                                       n_steps = self.n_step,
                                       windows_size = self.windows_size,
                                       n_channel = self.n_channel,
                                       overlap = self.overlap,
                                       get_data = False,
                                       get_delta_coded_data = True,
                                       get_beats_present=True)
            generators.append(gen)

        data_loader = utils.LoadDataFileShuffling(path_to_data = os.path.dirname(path_to_file),
                                                  batch_size = 1,
                                                  n_steps = N_STEP,
                                                  windows_size = WINDOWS_SIZE,
                                                  n_channel = N_CHANNELS,
                                                  overlap = OVERLAP,
                                                  get_data = False,
                                                  get_delta_coded_data = True,
                                                  get_beats_present = True,
                                                  verbose = VERBOSE)
        data_loader.generators = generators
        data_loader.batch_size = n_chunks

        #result = [np.zeros([0, len(REQUIRED_DISEASES)]) for i in range(len(generators))]
        result = [np.zeros([len(chunk['beats']), len(REQUIRED_DISEASES)]) for chunk in chunked_data]

        n_batches = (len(misc.get_channels(chunked_data[0])[0]) - self.overlap * self.windows_size) // (self.n_step * self.windows_size)
        forward_pass_time = 0
        for current_iter in tqdm(range(n_batches)):
            batch = data_loader.get_batch()
            feedDict = {self.inputX_delta_coded : batch['delta_coded_data'],
                        self.input_beats : (batch['beats_present'] > -0.5).astype(int),
                        self.keep_prob : 1}

            start_time = time.time()
            res = self.sess.run(self.predicted_events, feed_dict = feedDict) #b*s x len(REQUIRED_DISEASES)
            forward_pass_time = forward_pass_time + (time.time() - start_time)
            res = res.reshape([len(generators), N_STEP, len(REQUIRED_DISEASES)]) #b x s x len(REQUIRED_DISEASES)
            list_of_res = list(res)
            for i,r in enumerate(list_of_res):
                mask = batch['beats_present'][i,self.n_step] > -0.5 #b x s
                inds = (batch['beats_present'][i,self.n_step][mask] - 1).astype(int)
                result[i][inds,:] = r[mask,:]

        restored_data = utils.gathering_data_from_chunks(data, list_of_res = result, overlap=overlap, n_chunks=n_chunks)

        if path_to_predicted_beats is not None:
            np.save(path_to_predicted_beats, restored_data)
            print('\nfile saved ', path_to_predicted_beats)

        print('forward_pass_time = ', forward_pass_time)
        print('predicting_time = ', time.time() - predicting_time)
        return restored_data

# testing #####################################################################################################################
"""
path_to_file = '../data/test/AAO3CXJKEG.npy'
data = np.load(path_to_file).item()
n_chunks=128
overlap = 700 #in samples

chunked_data = utils.chunking_data(data)
"""





