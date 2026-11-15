# XPawSure AI

AI-powered canine skin disease preliminary screening system.

Target Diseases:
- Mange
- Allergic Dermatitis
- Bacterial Infection
- Fungal Infection
- Hot Spots

Tech Stack:
- Python
- TensorFlow
- OpenCV
- NumPy
- Pandas

## Model files

The production model **ships with this repo** so a fresh clone works out of the box:

- `models/fine_tuned_best_effnet_clean_5class_s42.keras` — EfficientNetB0, 5-class, 93.71% test accuracy (35 MB)
- `models/class_names_clean_5class_s42.json` — class order

Quick check after cloning (uses the bundled test set):
```
venv\Scripts\python.exe verify_5class_s42.py
```

**Not in the repo** (too large — GitHub 100 MB/file limit; back up separately):
- all other checkpoints in `models/` (alternate seeds, MobileNet/ResNet/DenseNet, ensembles)
- `compare/ckpt/*.keras` (model-comparison study checkpoints, up to 219 MB each)