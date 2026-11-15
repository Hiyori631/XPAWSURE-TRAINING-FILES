# XPawSure — Phase Tracking

## Objective
Improve classification accuracy for canine skin disease detection using MobileNetV2 / EfficientNetB0 + TensorFlow/Keras on merged public + local Philippine dataset.

## Datasets
### Original (leaked)
- `ProcessedDataset224_PublicLocal`: Train (329), Valid (62), Test (73)
- **31 images byte-for-byte identical across splits** → accuracy inflated

### Clean (deduplicated + merged with Datasets224new)
- `ProcessedDataset224_Clean`: Train (587), Valid (125), Test (130)
- MD5-hash deduplicated, zero cross-split overlap
- Merged 414 new images from Datasets224new (403 non-overlapping)
- Classes: Bacterial, Fungal, Hotspot, Mange

## Final Results (Clean Dataset)

### Best Model: `fine_tune_effnet.py` on `ProcessedDataset224_Clean`
| Metric | Frozen Phase | Fine-tune Phase |
|--------|:-----------:|:---------------:|
| Test Accuracy | **92.31%** | **92.31% ★** |

### Per-Class Breakdown (92.31% — EffNetB0 seed 42, clean data)
| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Bacterial | 0.86 | 0.89 | 0.87 |
| Fungal | 0.97 | 0.93 | 0.95 |
| Hotspot | 0.95 | 0.95 | 0.95 |
| Mange | 0.92 | 0.92 | 0.92 |

### Multi-Seed Results (EfficientNetB0 — Clean Dataset)
| Seed | Frozen Phase | Fine-Tune Phase |
|:----:|:-----------:|:---------------:|
| **42** | **92.31%** | **92.31% ★** |
| 7 | 92.31% | 92.31% |
| 123 | 90.77% | 90.00% |

## Historical Results (Leaked Dataset — for reference only)

### Best Model: `fine_tune_effnet.py` (EfficientNetB0 + SWA + custom class weights + richer aug)
| Metric | Frozen Phase | Fine-tune Phase |
|--------|:-----------:|:---------------:|
| Test Accuracy | **73.97%** | **76.71% ★** *(inflated by data leakage)* |

### All Experiments (Leaked Dataset)
| Experiment | Test Accuracy | Notes |
|-----------|:-----------:|-------|
| Baseline (no weights, basic aug) | 65.75% | Original best — MobileNetV2 |
| Custom weights + richer aug | **69.86%** | Previous best — weights (Fungal=2.5, Bacterial=2.0, Mange=1.3, Hotspot=1.0) + hue/saturation |
| **EfficientNetB0 + SWA (seed 42)** | **76.71%** | **★ OLD BEST** — EffNetB0 backbone + SWA averaging + 35 unfrozen layers |
| Richer aug (no weights) | 67.12% | Frozen phase only |
| Label Smoothing 0.05 | 60.27% | Hurt — conflicts with focal loss on small data |
| Full unfreeze (all 154) | 60.27% | Overfit |
| Mixup + Label Smoothing | ~52% | Over-regularized |
| Ensemble (5-fold) | 65.75% | No improvement |
| Ensemble + TTA | 63.01% | Worse |
| TTA (N=10) | 65.75% | No improvement |
| Oversampling (Fungal 2x) + SGD | 47.95% | Collapsed — Mange domination worsened; reverted |
| Recovery retrain (custom weights + rich aug) | 67.12% | Retrain from scratch after oversampling destroyed model |
| Two-stage unfreeze (35→50 layers, EffNet) | 71.23% | Regressed — overfitting from 130 total epochs |
| CutMix fine-tuning (EffNet, hard labels) | 63.01% | Regressed — CutMix conflicts with small dataset |
| Ensemble EffNet 3 seeds (42+7+123) | 73.97% | Worse than single best |

### Per-Class Breakdown (old best: 76.71% — leaked)
| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Bacterial | 0.71 | 0.62 | 0.67 |
| Fungal | 0.63 | 0.71 | 0.67 |
| Hotspot | 0.80 | 0.92 | 0.86 |
| Mange | 0.88 | 0.81 | 0.85 |

### Multi-Seed Results (MobileNetV2 — leaked)
| Seed | Frozen Phase | Fine-Tune Phase |
|:----:|:-----------:|:---------------:|
| 42 | 64.38% | 64.38% |
| **7** | **67.12%** | **69.86% ★** |
| 123 | **69.86%** | 69.86% |
| 999 | 67.12% | 64.38% |
| 777 | 67.12% | 67.12% |

### Multi-Seed Results (EfficientNetB0 — leaked)
| Seed | Frozen Phase | Fine-Tune Phase |
|:----:|:-----------:|:---------------:|
| **42** | **73.97%** | **76.71% ★** |
| 7 | 73.97% | 73.97% |
| 123 | 68.49% | 68.49% |

### Key Insight
EfficientNetB0 provides a ~6-7% boost over MobileNetV2 with the same training recipe. The top-35 unfrozen layers + SWA averaging add another ~3%. Gentle custom class weights + hue/saturation augmentation remain critical for this 329-sample dataset. Two-stage fine-tuning, CutMix, and ensemble all failed to improve over the single best seed 42 model — suggesting we've reached the practical ceiling for this dataset size.

---

## 5-Class Expansion (Allergic Dermatitis Added)

### Dataset
- `ProcessedDataset224_Clean_5class`: Train (641), Valid (136), Test (143)
- 78 unique Allergic Dermatitis images added (39 structured + 39 flat, zero hash overlap)
- Classes (alphabetical order by tf.keras): Allergic Dermatitis (0), Bacterial (1), Fungal (2), Hotspot (3), Mange (4)
- **Critical**: Class ordering is alphabetical, NOT the original manual mapping. Custom weights must match alphabetical indices.

### Custom Class Weights (corrected for alphabetical order)
```python
{0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}  # AD, Bacterial, Fungal, Hotspot, Mange
```

### Best Model: `fine_tune_effnet.py` on `ProcessedDataset224_Clean_5class` (Seed 42)
| Metric | Standard | TTA (N=10) | TTA Confident (≥60%) |
|--------|:--------:|:----------:|:--------------------:|
| Test Accuracy | **93.71%** | **95.10%** | **96.09%** (128/143 images) |

> **Verified 2026-10-20** — Standard 93.71% (134/143) independently re-confirmed by
> `verify_5class_s42.py` (read-only re-eval of `models/fine_tuned_best_effnet_clean_5class_s42.keras`
> on the 143-image test set). Per-class report matched the table below to the digit.
>
> **Warning — stale file:** `models/fine_tuned_cm_effnet_clean_5class_s42.png` is WRONG.
> It shows 127/143 (88.81%) from an earlier aborted run (PNG written 19:28, best model saved 19:54).
> 88.81% is actually seed 7's score. Do not read accuracy from that PNG.
>
> **TTA caveat:** 95.10% / 96.09% come from a console run only — no 5-class TTA output is saved
> on disk (the TTA scripts in this repo are still the old 4-class versions).
> 96.09% = accuracy on the 128 confident images only (123/128), 15 abstained.

### Per-Class Breakdown (Standard — 93.71%)
| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| Allergic Dermatitis | 0.86 | 0.92 | 0.89 | 13 |
| Bacterial | 0.92 | 0.89 | 0.91 | 27 |
| Fungal | 0.97 | 0.97 | 0.97 | 30 |
| Hotspot | 0.95 | 0.95 | 0.95 | 21 |
| Mange | 0.94 | 0.94 | 0.94 | 52 |

### Per-Class Breakdown (TTA Confident Only)
| Class | Accuracy | Below Threshold | Total |
|-------|:--------:|:---------------:|:-----:|
| Allergic Dermatitis | 92.3% (12/13) | 0 | 13 |
| Bacterial | 95.7% (22/23) | 4 | 27 |
| Fungal | 100.0% (29/29) | 1 | 30 |
| Hotspot | 93.8% (15/16) | 5 | 21 |
| Mange | 95.7% (45/47) | 5 | 52 |

### Multi-Seed Results (EfficientNetB0 — 5-Class Clean Dataset)
| Seed | Standard | TTA |
|:----:|:--------:|:---:|
| **42** | **93.71%** | **95.10% ★** |
| 7 | 88.81% | 92.31% |
| 123 | 90.91% | 93.01% |

### Model Files (5-class)
- `models/fine_tuned_best_effnet_clean_5class_s42.keras` — best model
- `models/class_names_clean_5class_s42.json` — class names
- `models/fine_tuning_log_effnet_clean_5class_s42.csv` — training log
- `verify_5class_s42.py` — read-only re-verification script (prints 134/143 = 93.71%, writes nothing)

## Phases

| Phase | File | Description | Status |
|-------|------|-------------|--------|
| 1 | `train.py` | Main training: focal loss, cosine decay, augmentation, dropout, L2 reg | ✅ Done |
| 2 | `fine_tune.py` | Fine-tune top 25 MobileNetV2 layers, BN frozen, CosineDecayRestarts LR | ✅ Done |
| 3 | `evaluate_model.py` | Full test evaluation with confusion matrix, classification report | ✅ Done |
| 4 | `fix_dataset.py` | Dataset restructuring: split local train, merged domains | ✅ Done |
| 5 | `predict.py` | Single-image prediction | ✅ Done |
| 6 | `test_verify.py` | TTA pipeline verification | ✅ Done |
| 7 | `tta_predict.py` | Test-Time Augmentation eval (N=10), compares std vs TTA | ✅ Done |
| 8a | `ensemble_train.py` | Stratified 5-fold cross-validation training | ✅ Done |
| 8b | `ensemble_predict.py` | Ensemble + TTA combined evaluation | ✅ Done |
| 9 | `inference.py` | Production inference pipeline (single/batch, ensemble+TTA) | ✅ Done |
| 10a | `train_effnet.py` | EfficientNetB0 frozen phase + SWA | ✅ Done |
| 10b | `fine_tune_effnet.py` | EfficientNetB0 fine-tune (top 35 layers) + SWA | ✅ Done |
| 11 | `run_multi_seed.py` | Multi-seed orchestrator for any pipeline | ✅ Done |
| 12 | `fine_tune_effnet_two_stage.py` | Two-stage EffNet fine-tune (35→50 layers) | ✅ Done (71.23%, regressed) |
| 13 | `fine_tune_effnet_cutmix.py` | EffNet + CutMix fine-tune | ✅ Done (63.01%, regressed) |
| 14 | `ensemble_effnet_seeds.py` | Ensemble of 3 EffNet seeds | ✅ Done (73.97%, regressed) |
| 15 | `build_clean_dataset.py` | MD5-deduplicated merge of PublicLocal + Datasets224new, 70/15/15 split | ✅ Done |
| 16 | Dataset upgrade | `ProcessedDataset224_Clean`: 587 train / 125 valid / 130 test (no leakage) | ✅ Done |
| 17 | `--dataset` arg | Added to `train_effnet.py`, `fine_tune_effnet.py`, `run_multi_seed.py` | ✅ Done |
| 18 | 5-class expansion | Added Allergic Dermatitis (78 images), 5-class clean dataset, corrected alphabetical class weights | ✅ Done |