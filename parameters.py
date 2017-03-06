parameters = {
    'sample_rate':175,
    'channels':['ES', 'AS', 'AI'], # input channels ['ES', 'AS
    'convert_to_channels':None, # None if no convertation need ['II', 'V1']
    'batch_size':64,
    'n_frames':5,
    'overlap':6,
    'rr':8,
    'n_hidden_RNN':128,
    'n_hidden_FC':256,
    'dropout':1,
    'L2':0.0001,
    'learn_rate':0.001,
    'use_delta_coding':True,
    'use_chunked_data':False,
    'verbose':True,
    'required_diseases':
        ['Atrial_PAC',
        'Atrial_Bigeminy',
        'Atrial_Trigeminy',
        'Atrial_Pair',
        'Atrial_Run',
        'Atrial_Late',
        'Atrial_Drop',
        'Atrial_Bradycardia',
        'Atrial_Tachycardia',
        'Atrial_Afib',
        'Ventricular_Signle_VE',
        'Ventricular_PVC',
        'Ventricular_Interp_PVC',
        'Ventricular_R_on_T',
        'Ventricular_Bigeminy',
        'Ventricular_Trigeminy',
        'Ventricular_Couplet',
        'Ventricular_Triplet',
        'Ventricular_VRun',
        'Ventricular_Late'],
    'all_diseases':
        ['?', '?', '?', '?', '?', '?', '?', '?', 'Other_Diary', '?', '?',
        '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
        'Other_N_and_V', 'Other_N_and_V_Late', 'Other_N_and_V_Pre',
        'Other_Noise_Minus', 'Other_Noise_Question', 'Other_Noise_Mul',
        'Other_ST_Episodes', '?', '?', '?', '?', '?', '?',
        'Ventricular_Late', '?', 'Ventricular_VRun', 'Ventricular_Triplet',
        'Ventricular_Couplet', 'Ventricular_Trigeminy',
        'Ventricular_Bigeminy', 'Ventricular_R_on_T',
        'Ventricular_Interp_PVC', 'Ventricular_PVC',
        'Ventricular_Signle_VE', '?', '?', '?', '?', '?', 'Atrial_Afib',
        '?', 'Atrial_Tachycardia', '?', 'Atrial_Bradycardia', 'Atrial_Drop',
        'Atrial_Late', '?', 'Atrial_Run', 'Atrial_Pair', 'Atrial_Trigeminy',
        'Atrial_Bigeminy', 'Atrial_PAC']
}

import numpy as np
a = parameters['all_diseases']
r = parameters['required_diseases']
disease_names = np.array(a, dtype=object)[np.in1d(a, r)]

parameters['disease_names'] = disease_names
    

if parameters['convert_to_channels'] is not None:
    assert parameters['channels'] == ['ES', 'AS', 'AI'], 'To convert to other channels need all ES, AS, AI channels'
parameters['n_channels'] = len(parameters['channels'])\
    if parameters['convert_to_channels'] is None\
    else len(parameters['convert_to_channels'])
