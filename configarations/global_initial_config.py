import torch 

# GLOBAL CONFIGARATIONS
SAMPLE_RATE = 16000
DURATION = 10.0
INPUT_NAME = None
PERRIFERAL_NAME = None
IS_TRACK_ID = True
TRANSFORMS = None
AUDIO_DIR = None
COMPONENTS = []
CSV_FILE = None
N_FFT = 512
HOP_LENGTH = 32
CACHE_DIR_PATH = None
DB_FILENAME = None
DB_FILE_PATH = None 
WAV_LENGTH = None
WINDOW_CPU = None

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
    cache_dir_path = None,
    db_filename = None,
    wav_length = None,
):
    global SAMPLE_RATE, DURATION, INPUT_NAME, PERRIFERAL_NAME, IS_TRACK_ID
    global TRANSFORMS, AUDIO_DIR, COMPONENTS, CSV_FILE,N_FFT,HOP_LENGTH
    global CACHE_DIR_PATH,DB_FILENAME,DB_FILE_PATH,WAV_LENGTH, WINDOW_CPU

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
        WINDOW_CPU = torch.hann_window(N_FFT, device="cpu")
    if hop_length is not None:
        HOP_LENGTH = hop_length
    if cache_dir_path != None:
        CACHE_DIR_PATH = cache_dir_path
    if db_filename != None:
        DB_FILENAME = db_filename
    if CACHE_DIR_PATH and DB_FILENAME != None:
        DB_FILE_PATH = CACHE_DIR_PATH/DB_FILENAME
    if wav_length != None:
        WAV_LENGTH = wav_length





#RANDOM-NOISE CONFIGARATIONS
NOISE_WAV_DIR = "./sample_noise"
NOISE_TENSOR_SAVE_DIR = "./sample_noise/pre_saved_tensors"
NOISE_TENSOR_NAME = "shuffled_big_raw_noise_tensor.pt"
CHUNK_DURATION_RECONSTRUCTED = 1
