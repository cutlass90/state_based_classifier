#data parameters
SAMPLE_RATE = 175
CHANNELS = ['ES', 'AS', 'AI'] # input channels ['ES', 'AS', 'AI']
CONVERT_TO_CHANNELS = None # None if no convertation need ['II', 'V1']

# classifier parameters
BATCH_SIZE = 64
N_BEATS = 5
OVERLAP = 2
N_HIDDEN_RNN = 64 #number of neurons in RNN
N_HIDDEN_FC = 64
DROPOUT = 1
L2 = 0.03
LEARN_RATE = 0.001

# Data Loader paremeters
USE_DELTA_CODING = True
USE_CHUNKED_DATA = False
VERBOSE = True

REQUIRED_DISEASES = [
    'Atrial_PAC',
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
    'Ventricular_Late'
    ]


ALL_DISEASES = ['?', '?', '?', '?', '?', '?', '?', '?', 'Other_Diary', '?', '?',
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

import numpy as np
disease_names = np.array(ALL_DISEASES, dtype=object)[np.in1d(ALL_DISEASES,
    REQUIRED_DISEASES)]
    

if CONVERT_TO_CHANNELS is not None:
    assert CHANNELS == ['ES', 'AS', 'AI'], 'To convert to other channels need all ES, AS, AI channels'
N_CHANNELS = len(CHANNELS) if CONVERT_TO_CHANNELS is None else len(CONVERT_TO_CHANNELS)
