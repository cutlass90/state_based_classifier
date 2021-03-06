import os

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import parameters as PARAM
import classifier_tools as utils
from classifier import Classifier



print('\n\n\n\t----==== Import parameters ====----')
with open('parameters.py', 'r') as param:
    print(param.read())

os.makedirs('summary/', exist_ok = True)



#path_to_train_data = '../../data/little/'
#path_to_train_data      = '../../data/train/chunked/'
path_to_train_data      = '../../../ECG_DATA/all/chunked_data/'
path_to_train_data = '../../data/small_set/'
path_eval_cost_data     = '../../../ECG_DATA/ECG_DATA_1000samples_2/test/'
path_to_predict_data    = path_to_train_data
path_to_predictions     = 'predictions/'
os.makedirs(path_to_predictions, exist_ok = True)
n_iter_train            = 9000
n_iter_eval             = 10000
save_model_every_n_iter = 10000
path_to_model = 'models/cl'


gen_params = dict(n_frames = PARAM['n_frames'],
                overlap = PARAM['overlap'],
                get_data = not(PARAM['use_delta_coding']),
                get_delta_coded_data = PARAM['use_delta_coding'],
                get_events = True,
                rr = PARAM['rr'])
"""
# Initialize data loader for training
data_loader = utils.LoadDataFileShuffling(batch_size=PARAM['batch_size'],
                                    path_to_data=path_to_train_data,
                                    gen=utils.step_generator,
                                    gen_params=gen_params,
                                    verbose=PARAM['verbose'])

# Train model
with Classifier(batch_size = PARAM['batch_size'],
                n_beats = PARAM['n_frames'],
                overlap = PARAM['overlap'],
                n_channel = PARAM['n_channels'],
                nHiddenRNN = PARAM['n_hidden_RNN'],
                nHiddenFC = PARAM['n_hidden_FC'],
                dropout = PARAM['dropout'],
                L2 = PARAM['L2'],
                do_train = True) as classifier:
    
    classifier.train_(data_loader = data_loader,
                         n_iter = n_iter_train,
                         save_model_every_n_iter = save_model_every_n_iter,
                         path_to_model = path_to_model) 
"""

# Predicting
#PARAM['use_chunked_data'] = False

#path_to_filenames = '/media/host1/Data/Work/ECG_DATA/all/valid_files/100.txt'
#with open(path_to_filenames, 'r') as file:
    #paths = file.read().splitlines()[:100]
paths = utils.find_files(path_to_predict_data, '*.npy')

"""
with Classifier(batch_size = 1,
                n_beats = PARAM['n_frames'],
                overlap = PARAM['overlap'],
                n_channel = PARAM['n_channels'],
                nHiddenRNN = PARAM['n_hidden_RNN'],
                nHiddenFC = PARAM['n_hidden_FC'],
                dropout = PARAM['dropout'],
                L2 = PARAM['L2'],
                do_train = False) as classifier:
    for path in paths:
        events = classifier.predicting_events(path_to_file = path,
            path_to_predicted_beats = path_to_predictions+os.path.basename(path)[:-4]+\
                "_events.npy",
            path_to_model = os.path.dirname(path_to_model))
"""

# create logs
true_l, pred_l = [], []
for path in paths:
    data = np.load(path).item()
    true_events = data['events'][:, np.in1d(data['disease_name'], PARAM['required_diseases'])]
    pred_events = np.load(path_to_predictions + os.path.basename(path)[:-4]+"_events.npy")

    pred_events_l = (pred_events > 0.5).astype(int)
    true_ev = np.hstack((true_events, (true_events.sum(1)[:,None]==0).astype(int)))
    pred_ev = np.hstack((pred_events_l, (pred_events_l.sum(1)[:,None]==0).astype(int)))

    true_labels = [np.nonzero(true_ev[r])[0][0] for r in range(true_events.shape[0])]
    pred_labels = [np.nonzero(pred_ev[r])[0][0] for r in range(pred_events.shape[0])]
    pred_l += pred_labels
    true_l += true_labels
    utils.plot_confusion_matrix(
        true_labels=true_labels + list(range(len(PARAM['disease_names']))),
        pred_labels=pred_labels + list(range(len(PARAM['disease_names']))),
        classes=np.insert(PARAM['disease_names'], len(PARAM['disease_names']), 'not_event'),
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues,
        save_path = path_to_predictions + os.path.basename(path)[:-4]+"_CM.jpg")
        #save_path = path_to_predictions + os.path.basename(path)[:-4]+"_CM.jpg")
    utils.save_log(
        path = path_to_predictions,
        file_name = os.path.basename(path)[:-4] + '.csv',
        diseases = data['disease_name'][np.in1d(data['disease_name'], PARAM['required_diseases'])],
        lbs = true_events,
        pred = pred_events,
        cost = np.arange(len(PARAM['required_diseases'])),
        threshold = 0.5)


# create summaty
data = np.load(paths[0]).item()
diseases=data['disease_name'][np.in1d(data['disease_name'], PARAM['required_diseases'])]
utils.save_summary(path=path_to_predictions, diseases=diseases)

utils.plot_confusion_matrix(
    true_labels=pred_l + list(range(len(PARAM['disease_names']))),
    pred_labels=true_l + list(range(len(PARAM['disease_names']))),
    classes=np.insert(PARAM['disease_names'], len(PARAM['disease_names']), 'not_event'),
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues,
    save_path = "summary_CM.jpg")


