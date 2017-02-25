import os

import tensorflow as tf
import numpy as np

from parameters import *
import classifier_tools as utils
from classifier import Classifier



print('\n\n\n\t----==== Import parameters ====----')
with open('parameters.py', 'r') as param:
    print(param.read())





#path_to_train_data = '../../data/little/'
#path_to_train_data      = '../../data/train/chunked/'
path_to_train_data      = '../../../ECG_DATA/all/chunked_data/'
path_eval_cost_data     = '../../../ECG_DATA/ECG_DATA_1000samples_2/test/'
path_to_predict_data    = path_to_train_data
path_to_predictions     = 'predictions/'
os.makedirs(path_to_predictions, exist_ok = True)
n_iter_train            = 14000
n_iter_eval             = 10000
save_model_every_n_iter = 15000
path_to_model = 'models/cl'


gen_params = dict(n_frames = N_BEATS,
                overlap = OVERLAP,
                get_data = False,
                get_delta_coded_data = True,
                get_events = True)

# Initialize data loader for training
data_loader = utils.LoadDataFileShuffling(batch_size=BATCH_SIZE,
                                    path_to_data=path_to_train_data,
                                    gen=utils.step_generator,
                                    gen_params=gen_params,
                                    verbose=True)



# Train model
with Classifier(batch_size = BATCH_SIZE,
                n_beats = N_BEATS,
                overlap = OVERLAP,
                n_channel = N_CHANNELS,
                nHiddenRNN = N_HIDDEN_RNN,
                nHiddenFC = N_HIDDEN_FC,
                dropout = DROPOUT,
                do_train = True) as classifier:
    
    classifier.train_(data_loader = data_loader,
                         n_iter = n_iter_train,
                         save_model_every_n_iter = save_model_every_n_iter,
                         path_to_model = path_to_model) 



path_to_filenames = '/media/host1/Data/Work/ECG_DATA/all/valid_files/100.txt'
with open(path_to_filenames, 'r') as file:
    paths = file.read().splitlines()[:100]
#paths = utils.find_files(path_to_predict_data, '*.npy')
# Predicting   
with Classifier(batch_size = 1,
                n_beats = N_BEATS,
                overlap = OVERLAP,
                n_channel = N_CHANNELS,
                nHiddenRNN = N_HIDDEN_RNN,
                nHiddenFC = N_HIDDEN_FC,
                dropout = DROPOUT,
                do_train = False) as classifier:
    for path in paths:
        events = classifier.predicting_events(path_to_file = path,
            path_to_predicted_beats = path_to_predictions + os.path.basename(path)[:-4]+"_events.npy",
            path_to_model = os.path.dirname(path_to_model))


# create logs
for path in paths:
    data = np.load(path).item()
    true_events = data['events'][:, np.in1d(data['disease_name'], REQUIRED_DISEASES)]
    pred_events = np.load(path_to_predictions + os.path.basename(path)[:-4]+"_events.npy")
    utils.save_log(
        path = path_to_predictions,
        file_name = os.path.basename(path)[:-4] + '.csv',
        diseases = data['disease_name'][np.in1d(data['disease_name'], REQUIRED_DISEASES)],
        lbs = true_events,
        pred = pred_events,
        cost = np.arange(len(REQUIRED_DISEASES)),
        threshold = 0.5)

# create summaty
data = np.load(paths[0]).item()
diseases=data['disease_name'][np.in1d(data['disease_name'], REQUIRED_DISEASES)]
utils.save_summary(path=path_to_predictions, diseases=diseases)


