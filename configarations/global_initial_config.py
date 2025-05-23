# GLOBAL CONFIGARATIONS
SAMPLE_RATE = 16000
DURATION = 5.0
INPUT_NAME = None
PERRIFERAL_NAME = None
IS_TRACK_ID = True
TRANSFORMS = None
AUDIO_DIR = None
COMPONENTS = []
CSV_FILE = None
N_FFT = 2048
HOP_LENGTH = 512

def update_config(
    sample_rate=None,
    duration=None,
    input_name=None,
    perriferal_name=None,
    is_track_id=None,
    transforms=None,
    audio_dir=None,
    components=None,
    csv_file=None,
    n_fft =None,
    hop_length = None,
):
    global SAMPLE_RATE, DURATION, INPUT_NAME, PERRIFERAL_NAME, IS_TRACK_ID
    global TRANSFORMS, AUDIO_DIR, COMPONENTS, CSV_FILE,N_FFT,HOPE_LENGTH

    if sample_rate is not None:
        SAMPLE_RATE = sample_rate
    if duration is not None:
        DURATION = duration
    if input_name is not None:
        INPUT_NAME = input_name
    if perriferal_name is not None:
        PERRIFERAL_NAME = perriferal_name
    if is_track_id is not None:
        IS_TRACK_ID = is_track_id
    if transforms is not None:
        TRANSFORMS = transforms
    if audio_dir is not None:
        AUDIO_DIR = audio_dir
    if components is not None:
        COMPONENTS = components
    if csv_file is not None:
        CSV_FILE = csv_file
    if n_fft is not None:
        N_FFT = n_fft
    if hop_length is not None:
        HOP_LENGTH = hop_length




#RANDOM-NOISE CONFIGARATIONS
NOISE_WAV_DIR = "./sample_noise"
NOISE_TENSOR_SAVE_DIR = "./sample_noise/pre_saved_tensors"
NOISE_TENSOR_NAME = "shuffled_big_raw_noise_tensor.pt"
CHUNK_DURATION_RECONSTRUCTED = 1
