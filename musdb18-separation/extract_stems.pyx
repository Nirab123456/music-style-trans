# cython: language_level=3
import os
import subprocess
from pathlib import Path
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def extract_stem(str input_file, int stream_index, str output_path):
    """
    Extract a single audio stem (specified by stream_index) from an MP4 file using FFmpeg.
    The output is saved in WAV format.
    """
    # Ensure that the output directory exists.
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    # Run FFmpeg with full path (adjust if your FFmpeg is elsewhere)
    subprocess.run([
        r"bin\ffmpeg.exe", "-y",
        "-i", input_file,
        "-map", f"0:a:{stream_index}",
        "-c:a", "pcm_s16le",
        output_path
    ], check=True)

def extract_all_stems(str input_file, str output_dir):
    """
    Extract all available audio stems from an MP4 file.
    This function uses FFprobe to determine the number of audio streams,
    and then extracts each stream using extract_stem().
    
    Returns a dictionary mapping stream index to the output file path.
    """
    cdef dict results = {}
    # Run ffprobe to count audio streams.
    cmd_probe = [r"D:\ffmpeg\bin\ffprobe.exe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=index", "-of", "csv=p=0", input_file]
    result = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
    cdef list streams = result.stdout.strip().split("\n")
    cdef int i
    cdef str output_path, stem_name

    # Use a default ordering for stem names.
    default_stems = ["mixture", "drums", "bass", "other_accompaniment", "vocals"]
    for i in range(len(streams)):
        if i < len(default_stems):
            stem_name = default_stems[i]
        else:
            stem_name = f"stem_{i}"
        output_path = os.path.join(output_dir, f"{stem_name}.wav")
        extract_stem(input_file, i, output_path)
        results[i] = output_path
    return results
