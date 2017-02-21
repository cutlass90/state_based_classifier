#data parameters
SAMPLE_RATE = 175
CHANNELS = ['ES', 'AS', 'AI'] # input channels ['ES', 'AS', 'AI']
CONVERT_TO_CHANNELS = None # None if no convertation need ['II', 'V1']

# classifier parameters
BATCH_SIZE = 1
N_BEATS = 10
OVERLAP = 5
N_HIDDEN_RNN = 16 #number of neurons in RNN
N_HIDDEN_FC = 64
DROPOUT = 0.9

# Data Loader paremeters
USE_DELTA_CODING = True
USE_CHUNKED_DATA = False
VERBOSE = True

REQUIRED_DISEASES = [
    #'Atrial_PAC',
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
    

if CONVERT_TO_CHANNELS is not None:
    assert CHANNELS == ['ES', 'AS', 'AI'], 'To convert to other channels need all ES, AS, AI channels'
N_CHANNELS = len(CHANNELS) if CONVERT_TO_CHANNELS is None else len(CONVERT_TO_CHANNELS)
