import os
import extract_stems  # import the Cython module

# Define the input MP4 stem file and output directory.
input_file = r"D:\MUSIC_STYLE_TRANSFER\musdb18\test\Al James - Schoolboy Facination.stem.mp4"
output_dir = "output_stems"

# Ensure the output directory exists.
os.makedirs(output_dir, exist_ok=True)

# Call the extraction function.
# If you want to extract all streams, we can use extract_all_stems (if defined)
results = extract_stems.extract_all_stems(input_file, output_dir)

print("Stems extracted successfully:")
for stream_index, file_path in results.items():
    print(f"Stream {stream_index}: {file_path}")
