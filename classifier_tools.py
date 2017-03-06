from __future__ import division, print_function

import os, fnmatch
import time
from random import shuffle
import functools

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm as tqdm
from sklearn.metrics import confusion_matrix

import misc
from parameters import parameters as PARAM

print('tools 26$$$$')


class LoadDataFileShuffling:

    def __init__(self,
                 batch_size,
                 path_to_data,
                 gen,
                 gen_params,
                 verbose = False):

        self.batch_size = batch_size
        self.path_to_data = path_to_data      
        self.verbose = verbose
        self.gen_params = gen_params
        self.current_list_of_data = []
        self.paths_to_data = find_files(path = path_to_data, file_type = '*.npy')
        shuffle(self.paths_to_data)
        
        self.n_epoch = 0
        self.n_batches = 0

        if verbose == True:
            print('Find ' + str(len(self.paths_to_data)) + ' files.')
        
        self.generators = [self.get_gen() for b in range(batch_size)]            
            
    ############################################################################
    def get_gen(self):
        if not(self.paths_to_data):
            print('Epoch was finished')
            self.n_epoch += 1
            self.paths_to_data = find_files(path = self.path_to_data,
                file_type = '*.npy')
            shuffle(self.paths_to_data)

        if PARAM['use_chunked_data']:
            if len(self.current_list_of_data) == 0:
                self.current_list_of_data = np.load(self.paths_to_data[0])
                shuffle(self.current_list_of_data)
                n_ch = np.random.randint(10,20)
                if len(self.current_list_of_data) >= n_ch:
                    self.current_list_of_data = self.current_list_of_data[:n_ch]
                if self.verbose:
                    pass
                    #print("\nFile " + self.paths_to_data[0] + " was loaded")
                self.paths_to_data.remove(self.paths_to_data[0])
            data = self.current_list_of_data[-1]
            self.current_list_of_data = self.current_list_of_data[:-1]
            ###### ETO KOSTILi, POTOMU, CHTO BUG V NAREZANIH DANNIH
            data['beats'] = data['beats'][:-1]
            ######
        else:
            if self.verbose:
                pass
                #print("\nFile " + self.paths_to_data[-1] + " was loaded")
            data = np.load(self.paths_to_data.pop()).item()
        
        gen =  step_generator(data, **self.gen_params)

        return gen
    
    ############################################################################
    def get_batch(self):
        batch = []

        for g, generator in enumerate(self.generators):
            n_attempts = 0
            while (n_attempts < 200):
                try:
                    batch.append(next(generator))
                    break
  
                except StopIteration:
                    generator = self.get_gen()
                    self.generators[g] = generator
                    n_attempts += 1

                if n_attempts > 190:
                    raise ValueError("Can't load 190 files in raw.")
            self.n_batches += 1
        preprocessed_batch = self.batch_preprocessing(batch)

        return preprocessed_batch

    ############################################################################
    def batch_preprocessing(self, batch):
        # batch is a list of entities that were returned from generator.
        # return events shape is [b*(n_frames+overlap), len(PARAM['required_diseases'])]
        # return normal_data shape is [b*(n_frames+overlap), x, n_channel]
        # return sequence_length shape is [b*(n_frames+overlap)]
        p_batch = {}
        tot_beats = self.gen_params['n_frames']+self.gen_params['overlap']

        if self.gen_params['get_data'] or self.gen_params['get_delta_coded_data']:
            p_batch['sequence_length'] = np.concatenate([d['sequence_length'] \
                for d in batch], 0)
            p_batch['sequence_length'] = p_batch['sequence_length'].astype(np.int32)
        else:
            None
        
        
        if self.gen_params['get_data']:
            n_channels = batch[0]['normal_data'].shape[2]
            data = np.zeros([self.batch_size*tot_beats,
                p_batch['sequence_length'].max(), n_channels])
            for i, b in enumerate(batch):
                s = i * tot_beats
                e = s + tot_beats
                data[s:e, :b['normal_data'].shape[1],:] = b['normal_data']
            p_batch['normal_data'] = data
        else:
            p_batch['normal_data'] = None
        

        if self.gen_params['get_delta_coded_data']:
            n_channels = batch[0]['delta_coded_data'].shape[2]
            data = np.zeros([self.batch_size*tot_beats,
                p_batch['sequence_length'].max(), n_channels])
            for i, b in enumerate(batch):
                s = i * tot_beats
                e = s + tot_beats
                data[s:e, :b['delta_coded_data'].shape[1],:] = b['delta_coded_data']
            p_batch['delta_coded_data'] = data
        else:
            p_batch['delta_coded_data'] = None


        p_batch['events'] = np.concatenate(
            [d['events'] for d in batch], 0) \
        if self.gen_params['get_events'] else None
        
        
        if self.gen_params['get_events']:
            mask = np.in1d(batch[0]['disease_name'], PARAM['required_diseases'])
            p_batch['events'] = p_batch['events'][:, mask]
            #a = ~np.in1d(PARAM['required_diseases'], batch['disease_name'][mask])
            #print(np.array(PARAM['required_diseases'])[a])
            assert len(PARAM['required_diseases']) == mask.sum(), \
            'Some of requierd diseases not found. Check REQUIRED_DISEASES.'
        
        return p_batch

################################################################################
def find_files(path, file_type):
    #find all files of type file_type in directory and subdirectory path
    #return a list of sort path
    found_files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, file_type):
            found_files.append(os.path.join(root, filename))
    found_files.sort()
    
    return found_files

################################################################################
def XavierRandomMatrixInitializer(in_dim, out_dim, constant=1):
	w = constant * np.sqrt(6.0 / (in_dim + out_dim))
	return tf.random_uniform_initializer(minval=-w, maxval=w, dtype=tf.float32)

################################################################################
#@profile
def step_generator(data,
                   n_frames = 10,
                   overlap = 5,
                   get_data = False,
                   get_delta_coded_data = False,
                   get_events = False):
    
    #---------------------------------------------------------------------------
    def format_data(channels, start_beat, end_beat):
        # padded data shape [n_frames+overlap, max_len, len(channels)]
        sequence_length = np.empty([0], np.int8)
        channels_part_list = []
        for b in range(start_beat, end_beat):
            channels_part = np.concatenate([np.expand_dims(
                channel[data['beats'][b]:data['beats'][b+1]], 1)\
                for channel in channels], 1) #h x c (where h is variable value)
            channels_part_list.append(channels_part) # list len n_frames+overlap
                #of arrays h x c (where h is variable value)

            sequence_length = np.append(sequence_length, channels_part.shape[0])
        max_len = sequence_length.max()
        padded_data = np.zeros([n_frames+overlap, max_len, len(channels)], np.float16)
        for i, channel_part in enumerate(channels_part_list):
            padded_data[i, 0:channel_part.shape[0], :] = channel_part

        return padded_data, sequence_length
    #---------------------------------------------------------------------------

    # channels converting
    channels = misc.get_channels(data)
    if PARAM['convert_to_channels'] is not None:
        channels = misc.convert_channels_from_easi(channels, PARAM['convert_to_channels'])
    
    n_batches = (data['beats'].shape[0] - overlap) // n_frames - 1

    if get_delta_coded_data:
        channels_coded = [np.hstack([[0], np.ediff1d(channel)]).astype(np.float16)\
            for channel in channels]

    for current_batch in range(n_batches):
        yield_res = {'normal_data':None, 'delta_coded_data':None, 'events':None,
            'disease_name':data['disease_name'], 'sequence_length':None}

        start_beat = current_batch*(n_frames)
        end_beat = start_beat + n_frames + overlap
        
        if get_data:
            yield_res['normal_data'], yield_res['sequence_length'] = format_data(
                channels, start_beat, end_beat)

        if get_delta_coded_data:
            yield_res['delta_coded_data'], yield_res['sequence_length'] = format_data(
                channels_coded, start_beat, end_beat)

        if get_events:
            yield_res['events'] = data['events'][start_beat:end_beat,:]

        yield_res['sequence_length'] = yield_res['sequence_length'].astype(np.int32)

        yield yield_res

          
#----------------------------------------------------------------------------------------
def metrics(matrix):

	"""Computes metrics given a confusion matrix.
	
	Args:

		matrix: confusion matrix: TN FP
								  FN TP

	Returns: numpy array containing metrics.                          

	"""
	
	tp         = matrix[1][1]
	tn         = matrix[0][0]
	fp         = matrix[0][1]
	fn         = matrix[1][0]  
	num_events = tp + fn
	accuracy   = (tp + tn)/(tp + tn + fp + fn)
	precision  = tp/(tp + fp) if tp + fp > 0 else -1
	recall     = tp/(tp + fn) if tp + fn > 0 else -1
	fscore     = 2*precision*recall/(precision + recall) if precision + recall > 0 else -1
					
	return np.array([tp, tn, fp, fn, num_events, accuracy, precision, recall, fscore])

#-----------------------------------------------------------------------
def save_log(path, file_name, diseases, lbs, pred, cost, threshold):

	"""Given labels and prediction, evaluates metrics and saves results to the csv file

	Args:

		path: directory to save the log.
		epoch: epoch number.
		fl: path to current file.
		diseases: list of diseases.
		lbs: true labels.
		pred: predicted labels.
		cost: cost function. Must have the same len as diseases.
		threshold: threshold for sigmoidal prediction.

	Saves metrics to /path/file_name/
	""" 

	cost = np.reshape(cost, [len(diseases), 1])
	pred = (pred > threshold)
	scores = np.array([metrics(confusion_matrix(l, p, labels=[0, 1])) for l, p in zip(lbs.T, pred.T)])
	scores = np.hstack([scores, cost])
	
	names = ['{:^12}'.format(n) for n in ['tp', 'tn', 'fp', 'fn', 'num_events',
                                                'accuracy', 'precision', 'recall', 'fscore', 'cost']]
	
	df = pd.DataFrame(scores, diseases, names)
	p = os.path.join(path, file_name)
	os.makedirs(path, exist_ok=True)
	df.to_csv(p, sep='\t', float_format='%.3f')

#-----------------------------------------------------------------------
def save_summary(path, diseases):

	"""Computes summarized results across all .csv files in path directory for given epoch.

	Args:

		path: directory to search for csv files.
		epoch: epoch number.
		diseases: list of diseases.

	Saves summarized csv to the parent directory of path.
	"""
	
	files = find_files(path, '*.csv')
	
	dframes         = [pd.read_csv(f, index_col=0, sep='\t') for f in files]
	summary_frame   = functools.reduce(lambda x, y: y + x, dframes)
	names           = list(summary_frame.keys())

	cost    = np.reshape(summary_frame[names[-1]].as_matrix()/len(files), (len(diseases), 1))
	scores  = np.array([metrics([[tn, fp], [fn, tp]]) for (tp, tn, fp, fn) in 
										summary_frame[names[:4]].as_matrix()])   
	scores  = np.hstack([scores, cost])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
	
	#head, tail = os.path.split(path)
	averaged_frame  = pd.DataFrame(scores, diseases, names)

	#maybe_create_dirs(os.path.join(head, 'summary_' + tail))

	averaged_frame.to_csv(path+'summary.csv', sep='\t', float_format='%.3f')
	
	print('\nLogs saved.\n')

def chunking_data(data, n_chunks=128, overlap = 700):
    # |--------------len_of_chunk--------------|
    # |****************************|***********|
    # |                            |--overlap--|
    channels = misc.get_channels(data)
    len_of_chunk = (len(channels[0])-overlap)//n_chunks + 1 + overlap
    padding_size = (len_of_chunk - overlap)*n_chunks + overlap - len(channels[0])
    channels = [np.concatenate((channel, np.linspace(channel[-1], 0, padding_size)), axis = 0) for channel in channels]

    list_of_cunks = []
    for c in range(n_chunks):
        chunked_data = data.copy()
        chunk_begin = c*(len_of_chunk - overlap)
        chunk_end = chunk_begin + len_of_chunk
        
        chanked_channels = [channel[chunk_begin:chunk_end] for channel in channels]
        chunked_data = misc.write_channels(chunked_data, chanked_channels)

        inds = (data['beats']>=chunk_begin) & (data['beats']<chunk_end)
        chunked_data['beats'] = data['beats'][inds]
        chunked_data['beats'] = chunked_data['beats'] - chunk_begin

        chunked_data['events'] = data['events'][inds,:]
        list_of_cunks.append(chunked_data)

    return list_of_cunks

def gathering_data_from_chunks(data, list_of_res, overlap=700, n_chunks=32):
    predicted_events = list_of_res[0]
    channels = misc.get_channels(data)
    len_of_chunk = (len(channels[0])-overlap)//n_chunks + 1 + overlap

    for c in range(1, n_chunks):
        chunk_begin = c*(len_of_chunk - overlap)

        start_ind = np.sum((data['beats']>=chunk_begin) & (data['beats']<chunk_begin + overlap))
        predicted_events = np.concatenate((predicted_events, list_of_res[c][start_ind:,:]), 0)

    assert data['events'].shape[0] == predicted_events.shape[0], 'Original shape not equal reconstarct shape {0} != {1}'.format(data['events'].shape[0], predicted_events.shape[0])
    return predicted_events

#######################################################################################
#testing
"""
import sys
sys.path.append('../../Preprocessing/')
import Preprocessing_v2 as pre

data = np.load('../../data/little/AAO1CMED2K865.npy').item()
pre.view_beat_data(data, 0 , 13, plot_events=True)



gen_params = dict(n_frames = 5,
                overlap = 3,
                get_data = True,
                get_delta_coded_data = True,
                get_events = True) 

data_loader = LoadDataFileShuffling(batch_size=1,
                                    path_to_data='/media/nazar/DATA/Sapiens/ICG/data/little/',
                                    gen=step_generator,
                                    gen_params=gen_params,
                                    verbose=True)
REQUIRED_DISEASES = np.asarray(REQUIRED_DISEASES, dtype=object)
REQUIRED_DISEASES = data['disease_name'][np.in1d(data['disease_name'], REQUIRED_DISEASES)]

a = 0
while True:
    b = data_loader.get_batch()
    print('\n new batch')
    for i in range(b['events'].shape[0]):
        ind = b['events'][i,...] == 1
        print(REQUIRED_DISEASES[ind])
        plt.plot(b['delta_coded_data'][i,:,0])
        plt.show()
    input(a)
"""


"""
start_time = time.time()
while data_loader.n_epoch == 0:
    b = data_loader.get_batch()
print("Time  --- %s seconds ---" % (time.time() - start_time))
"""


"""
path_to_file = '../data/test/AAO3CXJKEG.npy'
data = np.load(path_to_file).item()
n_chunks=8
overlap = 700
list_of_res = []

predicted_events = list_of_res[0]
channels = misc.get_channels(data)
len_of_chunk = (len(channels[0])-overlap)//n_chunks + 1 + overlap

for c in range(1, n_chunks):
    chunk_begin = c*(len_of_chunk - overlap)
    chunk_end = chunk_begin + len_of_chunk

    start_ind = np.sum((data['beats']>=chunk_begin) & (data['beats']<chunk_begin + overlap))
    predicted_events = np.concatenate((predicted_events, list_of_res[c][start_ind:,:]), 0)

assert data['events'].shape == predicted_events.shape, 'Original shape not equal reconstarct shape {0} != {1}'.format(data['events'].shape, predicted_events.shape)
"""


"""
data = np.load('/media/nazar/DATA/Sapiens/ICG/data/test/AAO3CXJKEG.npy').item()
import sys
sys.path.append('../../Preprocessing/')
import Preprocessing_v2 as pre
pre.view_beat_data(data, 0 , 4)
gen = step_generator(data,
               n_frames = 5,
               overlap = 2,
               get_data = True,
               get_delta_coded_data = True,
               get_events = True)

b = next(gen)
"""

"""
start_time = time.time()
while True:
    try:
        b = next(gen)
    except StopIteration:
        break
print("Time  --- %s seconds ---" % (time.time() - start_time))
"""


"""
data_loader = LoadDataFileShuffling(
                 batch_size = 1,
                 path_to_data = '/media/nazar/DATA/Sapiens/ICG/data/test/',
                 n_steps = 10,
                 windows_size = 35,
                 n_channel = 3,
                 overlap = 10,
                 target_shift = 0,
                 skip_noise = False,
                 get_data = False,
                 get_delta_coded_data = True,
                 get_events = False,
                 get_energy_mask = False,
                 get_offsets = False,
                 get_beats = False,
                 get_dist = False,
                 get_ndist = False,
                 get_beats_present = True,
                 verbose = False)
    

batch = data_loader.get_batch()
"""


"""
data = np.load('/media/nazar/DATA/Sapiens/ICG/data/train/chunked/AAO1CMED2K0.npy').item()

events = data['events'][:,8:10]
REQUIRED_DISEASES = ['Atrial PAC', 'Ventricular_PVC']

save_log(path='test_metrics', epoch = 1, fl = 'fl', diseases = REQUIRED_DISEASES, lbs = events, pred = events, cost = [1,2], threshold = 0.5)
"""
