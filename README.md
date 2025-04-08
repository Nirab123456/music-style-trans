# 🎵 music-style-trans

Transfer the style or pattern from one group of music to another using audio separation and transformation techniques.

---

## 📦 Environment Setup

Create a new conda environment:

```bash
conda create -n sml python=3.12
conda activate sml
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🎧 MUSDB18 Dataset Parsing

To parse the MUSDB18 dataset:

1. Navigate to the parser directory:

   ```bash
   cd musdb18-separation
   ```

2. Copy the appropriate FFmpeg binary for your system into the `bin` folder (if not already included).

3. Build the Cython-based parser:

   ```bash
   python setup.py build_ext --inplace
   ```

4. Run the MUSDB18 separator:

   ```bash
   python musdb18_convert.py
   ```

This will extract and organize the audio stems for use in your music style transfer pipeline.

