# cython: language_level=3
import subprocess
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def extract_stems(str input_file, str output_dir):
    cdef list stem_names = ["vocals", "drums", "bass", "other"]
    cdef int i
    cdef str stem_name, output_path

    for i in range(len(stem_names)):
        stem_name = stem_names[i]
        output_path = f"{output_dir}/{stem_name}.wav"

        subprocess.run([
            r"bin\ffmpeg.exe", "-y",
            "-i", input_file,
            "-map", f"0:a:{i + 1}",  # skip 0:a:0 if it's mixture
            "-c:a", "pcm_s16le",
            output_path
        ], check=True)
