import extract_stems
import os

input_file = r"D:\MUSIC_STYLE_TRANSFER\musdb18\test\Al James - Schoolboy Facination.stem.mp4"
output_dir = "output_stems"

os.makedirs(output_dir, exist_ok=True)

extract_stems.extract_stems(input_file, output_dir)
print("Stems extracted successfully.")
