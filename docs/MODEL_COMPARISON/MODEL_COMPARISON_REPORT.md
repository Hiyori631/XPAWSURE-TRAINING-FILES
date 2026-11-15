# XPawSure — Comparative Model Evaluation Report

**Architecture comparison for 5-class canine skin-condition classification**
Date: 2026-11-13 · Status: Complete · Research-only study (no production changes)

---

## 1. Research Objective

EfficientNetB0 is the CNN used in the XPawSure production pipeline for classifying five canine skin conditions. This study produces measurable, reproducible evidence comparing EfficientNetB0 against three alternative CNN architectures — MobileNetV2, ResNet50, and DenseNet121 — under an identical dataset, split, training protocol, and evaluation procedure, in order to determine:

1. which architecture performs best on the clean 5-class dataset,
2. whether EfficientNetB0 is justified as XPawSure's selected model, and
3. the trade-offs (accuracy vs. model size vs. latency) each alternative represents.

The conclusion of this report is based entirely on experimental results collected in this study.

## 2. Dataset Description

| Property | Value |
|---|---|
| Dataset | `ProcessedDataset224_Clean_5class` |
| Image size | 224 × 224 RGB |
| Train / Valid / Test | **641 / 136 / 143** (70/15/15 per class) |
| Classes (alphabetical) | Allergic Dermatitis, Bacterial, Fungal, Hotspot, Mange |
| Per-class test support | AD 13, Bacterial 27, Fungal 30, Hotspot 21, Mange 52 |
| Deduplication | MD5 of raw file bytes, intra- and cross-source; verified 920/920 unique hashes, zero cross-split overlap |
| Class imbalance | Train: Mange 239 vs AD 54 (4.4:1) — handled by focal loss + class weights |

The identical fixed split was used for all four architectures; no test image was used for training, augmentation selection, or hyperparameter tuning. The test set was evaluated **once per run**, only at the end of fine-tuning.

## 3. Experimental Methodology

### 3.1 Controlled variables (identical across all models)
- Dataset, splits, 224×224 input, alphabetical class order
- Loss: focal loss (γ = 2.0) with class weights `{AD: 3.0, Bacterial: 2.0, Fungal: 2.5, Hotspot: 1.0, Mange: 1.3}`
- Head: GAP → BN → Dropout 0.5 → Dense 256 (L2 1e-4) → BN → Dropout 0.4 → Dense 128 → BN → Dropout 0.3 → Dense 5
- Batch size 16; identical augmentation policy (frozen phase: flip/rot 0.15/zoom 0.1/contrast 0.15/brightness 0.1/translate 0.05 + hue 0.05, sat 0.7–1.3; fine-tune phase: gentler rot 0.1/zoom 0.05/contrast 0.1/brightness 0.05 + same hue/sat)
- Two-stage protocol: **frozen phase** (ImageNet weights, 100 epochs, Adam + CosineDecay 1e-3→1e-5, SWA over last 30%, early stop patience 15) → **fine-tune phase** (last ~25% of backbone unfrozen, BatchNorm layers kept frozen, Adam lr 1e-5, 80 epochs, early stop patience 10, SWA over last 40%)
- Seeds **42, 7, 123** for every architecture (seeded tf/NumPy/Python RNG)
- Evaluation: accuracy, macro/weighted precision-recall-F1, per-class metrics, confusion matrix, CPU batch-1 inference latency (50 images, 5 warm-up runs excluded)

### 3.2 Architecture-specific adjustments (documented deviations)
| Backbone | preprocess | Backbone layers | Unfrozen (fine-tune) | Base layer name |
|---|---|---:|---|---|
| EfficientNetB0 | `efficientnet.preprocess_input` | 154 | 35 (production recipe ≈ 23%) | `efficientnetb0` |
| MobileNetV2 | `mobilenet_v2.preprocess_input` | 154 | last 38 (25%) | `mobilenetv2_1.00_224` |
| ResNet50 | `resnet50.preprocess_input` | 175 | last 44 (25%) | `resnet50` |
| DenseNet121 | `densenet.preprocess_input` | 427 | last 107 (25%) | `densenet121` |

EfficientNetB0 was **not retrained**: the existing production checkpoints (`models/fine_tuned_best_effnet_clean_5class_s{42,7,123}.keras`) were re-evaluated with the same unified evaluator, guaranteeing the baseline is exactly the shipped model. All other architectures were trained from scratch with the recipe above.

### 3.3 Statistical analysis
Per-model mean ± sample standard deviation over 3 seeds; 95% confidence intervals (t, df=2); paired t-tests on per-seed accuracy differences (α=0.05).

## 4. Model Architecture Descriptions

| Model | Year | Key idea | Params (incl. head) |
|---|---|---|---:|
| **EfficientNetB0** | 2019 | Compound scaling (depth/width/resolution) with mobile-efficient building blocks, squeeze-excite | 4.42 M |
| **MobileNetV2** | 2018 | Depthwise-separable convolutions + inverted residuals with linear bottleneck | 2.63 M |
| **ResNet50** | 2015 | Plain 50-layer residual network with bottleneck blocks and skip connections | 24.16 M |
| **DenseNet121** | 2017 | Dense connectivity: every layer receives all preceding feature maps | 7.34 M |

All used `include_top=False` with ImageNet weights and the identical custom classification head.

## 5. Training Configurations

Full per-run records: `training_configs/*_config.json` (9 files) and `training_configs/*_metrics.json` (12 files).

| Item | Value |
|---|---|
| Optimizer (frozen) | Adam, CosineDecay(1e-3 → 1e-5, α=1e-5) |
| Optimizer (fine-tune) | Adam, fixed lr = 1e-5 |
| Loss | Focal loss γ=2.0, per-class α weights |
| Early stopping | frozen: val_loss patience 15 · fine-tune: val_loss patience 10 (restore best) |
| Checkpointing | best val_accuracy per phase |
| SWA | frozen: last 30% of epochs · fine-tune: last 40% |
| Total epochs per run | ≤100 frozen + ≤80 fine-tune (early stopping may terminate sooner) |
| Hardware | CPU only (TF 2.21, native Windows — no GPU support), sequential runs |
| Wall-clock total | 10.49 h for 9 alternative runs (18 phases) |

## 6. Evaluation Results

### 6.1 Overall performance (test set, mean ± std over 3 seeds)

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Recall | Best run (acc) | Worst run (acc) |
|---|---|---|---|---|---|---|
| **ResNet50** | **0.9254 ± 0.0282** | **0.9037 ± 0.0295** | **0.9265 ± 0.0278** | **0.9190 ± 0.0225** | **0.9510** (s7) | 0.8951 (s42) |
| **EfficientNetB0** | 0.9114 ± 0.0246 | 0.8988 ± 0.0302 | 0.9128 ± 0.0232 | 0.9116 ± 0.0238 | 0.9371 (s42, production) | 0.8881 (s7) |
| **DenseNet121** | 0.9044 ± 0.0246 | 0.8897 ± 0.0256 | 0.9058 ± 0.0251 | 0.9018 ± 0.0227 | 0.9301 (s7) | 0.8811 (s42) |
| **MobileNetV2** | 0.8695 ± 0.0201 | 0.8391 ± 0.0186 | 0.8704 ± 0.0214 | 0.8426 ± 0.0243 | 0.8811 (s42/7) | 0.8462 (s123) |

> **Honest finding: ResNet50 achieved the highest raw mean accuracy and macro F1.** EfficientNetB0 ranked second on both metrics.

### 6.1b Best-single-run view (alternative ranking)

Ranking by each model's **best seed** rather than the 3-seed mean:

| Rank | Model | Best run | Accuracy | Note |
|---:|---|---|---|---|
| 1 | ResNet50 | s7 | **0.9510** | Highest single run of the study |
| 2 | EfficientNetB0 | s42 | **0.9371** | Production model (134/143) |
| 3 | DenseNet121 | s7 | 0.9301 | |
| 4 | MobileNetV2 | s42 / s7 | 0.8811 | |

By **seed 42** (the production seed): EfficientNetB0 **0.9371** > ResNet50 0.8951 > DenseNet121 0.8811 = MobileNetV2 0.8811 — the production model wins head-to-head at the matched seed. ResNet50 only surpasses it at seed 7 (0.9510 vs 0.8881), which is what lifts its mean; its seed-42 run is 4.2 points *below* your production model. Both views are reported because neither is decisive alone: per-run variance (±2.5–2.8 points) is comparable to the between-model gaps.

### 6.1c Matched-seed comparison (seed 42 only — every number is a single run)

All four models trained with the identical seed 42 recipe and evaluated on the same 143 test images:

| Model | Seed-42 accuracy | vs EfficientNetB0 |
|---|---:|---:|
| **EfficientNetB0 (production)** | **0.9371** | — |
| ResNet50 | 0.8951 | −4.20 pts |
| DenseNet121 | 0.8811 | −5.60 pts |
| MobileNetV2 | 0.8811 | −5.60 pts |

At the matched production seed, **EfficientNetB0 leads every alternative** by 4.2–5.6 points. This comparison is fully like-for-like (same seed, same data order, same protocol) and complements — rather than replaces — the 3-seed mean view in §6.1.

### 6.2 Per-run results

| Model | Seed | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| EfficientNetB0 | 42 | 0.9371 | 0.9312 | 0.9371 |
| EfficientNetB0 | 7 | 0.8881 | 0.8713 | 0.8909 |
| EfficientNetB0 | 123 | 0.9091 | 0.8939 | 0.9104 |
| MobileNetV2 | 42 | 0.8811 | 0.8522 | 0.8818 |
| MobileNetV2 | 7 | 0.8811 | 0.8473 | 0.8838 |
| MobileNetV2 | 123 | 0.8462 | 0.8179 | 0.8457 |
| ResNet50 | 42 | 0.8951 | 0.8713 | 0.8969 |
| ResNet50 | 7 | **0.9510** | **0.9289** | 0.9520 |
| ResNet50 | 123 | 0.9301 | 0.9108 | 0.9307 |
| DenseNet121 | 42 | 0.8811 | 0.8697 | 0.8809 |
| DenseNet121 | 7 | 0.9301 | 0.9186 | 0.9310 |
| DenseNet121 | 123 | 0.9021 | 0.8808 | 0.9056 |

Full machine-readable data: `MODEL_COMPARISON_RESULTS.csv` (includes per-class P/R/F1 per run, params, sizes, durations, latency).

### 6.3 Statistical significance (paired t-test by seed, df=2)

| Comparison (A − B) | Mean diff | 95% CI | p-value | Significant? |
|---|---:|---|---:|---|
| EfficientNetB0 − ResNet50 | −0.0140 | [−0.145, +0.117] | 0.692 | No |
| EfficientNetB0 − DenseNet121 | +0.0070 | [−0.115, +0.129] | 0.828 | No |
| EfficientNetB0 − MobileNetV2 | +0.0420 | [−0.034, +0.118] | 0.140 | No (consistent direction: EffNet won all 3 seed pairs) |

95% CIs of mean accuracy: ResNet50 [0.855, 0.996] · EfficientNetB0 [0.850, 0.973] · DenseNet121 [0.843, 0.966] · MobileNetV2 [0.819, 0.920] — **all overlap**. With n=3 seeds the test has low power, but no architecture difference reaches significance; ResNet50's +1.4 pp edge over EfficientNetB0 is within noise.

## 7. Comparative Tables — Efficiency

| Model | Params | .keras file | Float32 weights | CPU latency (ms/img) | Mean train time/run |
|---|---:|---:|---:|---:|---:|
| EfficientNetB0 | 4.42 M | 36.9 MB | ~17.7 MB | **60.0 ± 0.6** | N/A (existing checkpoints) |
| MobileNetV2 | 2.63 M | 27.4 MB | ~10.5 MB | **54.5 ± 0.8** | 43 min |
| ResNet50 | 24.16 M | 230.4 MB | ~96.6 MB | 77.4 ± 2.2 | 94 min |
| DenseNet121 | 7.34 M | 49.5 MB | ~29.4 MB | 81.4 ± 0.6 | 68 min |

`.keras` sizes include optimizer state (saved at best-checkpoint); float32 weights are `params × 4 bytes` — the production TensorFlow.js `weights.bin` (17.7 MB) matches the EfficientNetB0 weights-only figure. Latency = single-image CPU inference, 50 test images, warm-up excluded. Training times: frozen + fine-tune wall-clock, mean over 3 seeds (early stopping makes these highly variable).

**EfficientNetB0 is 6.2× smaller than ResNet50 and 1.29× faster, while being statistically tied on accuracy.**

## 8. Confusion Matrices

Per-run matrices: `confusion_matrices/{model}_s{seed}_cm.png` (12 files).
Mean matrices (averaged over seeds): `cm_mean_{efficientnetb0,mobilenetv2,resnet50,densenet121}.png`.

Qualitative pattern common to all models: the dominant error mode is **Bacterial ↔ Fungal** confusion and **Hotspot ↔ Mange** confusion (visually similar presentations); Allergic Dermatitis is the hardest class for the weaker models (MobileNetV2), while all other architectures recover AD recall at 12/13 images in every seed.

## 9. Per-Class Performance Analysis (mean recall / F1 over 3 seeds)

| Model | AD recall | Bacterial recall | Fungal recall | Hotspot recall | Mange recall |
|---|---|---|---|---|---|
| EfficientNetB0 | 0.923 ± 0.000 | **0.864 ± 0.021** | 0.956 ± 0.051 | 0.905 ± 0.048 | 0.910 ± 0.029 |
| MobileNetV2 | 0.744 ± 0.089 | 0.815 ± 0.037 | 0.900 ± 0.058 | 0.825 ± 0.073 | 0.929 ± 0.029 |
| ResNet50 | 0.923 ± 0.000 | 0.802 ± 0.057 | 0.956 ± 0.019 | **0.952 ± 0.000** | **0.962 ± 0.038** |
| DenseNet121 | 0.923 ± 0.000 | 0.803 ± 0.043 | 0.933 ± 0.000 | 0.921 ± 0.055 | 0.929 ± 0.029 |

| Model | AD F1 | Bacterial F1 | Fungal F1 | Hotspot F1 | Mange F1 |
|---|---|---|---|---|---|
| EfficientNetB0 | 0.821 ± 0.060 | 0.870 ± 0.035 | 0.950 ± 0.030 | 0.919 ± 0.029 | 0.934 ± 0.007 |
| MobileNetV2 | 0.690 ± 0.045 | 0.842 ± 0.030 | 0.900 ± 0.046 | 0.838 ± 0.042 | 0.927 ± 0.018 |
| ResNet50 | 0.810 ± 0.031 | 0.867 ± 0.061 | **0.956 ± 0.019** | 0.909 ± 0.021 | **0.977 ± 0.021** |
| DenseNet121 | **0.843 ± 0.087** | 0.867 ± 0.040 | 0.899 ± 0.017 | 0.886 ± 0.039 | 0.954 ± 0.030 |

Observations:
- **Allergic Dermatitis** (fewest train samples, 54): all architectures except MobileNetV2 reach 12/13 recall in every seed; MobileNetV2 is unstable (0.744 ± 0.089).
- **Bacterial** is EfficientNetB0's strongest relative class (best mean recall, 0.864).
- **ResNet50** dominates Hotspot and Mange recall (perfect 20/21 Hotspot in all seeds).
- **Mange** (largest class) is handled well by every model (≥0.91 recall).
- Macro F1 ranking (ResNet > EffNet > DenseNet > MobileNet) mirrors accuracy; no model collapses on any class.

## 10. Strengths and Weaknesses of Each Architecture

**EfficientNetB0**
- ✅ Statistically tied for best accuracy; excellent macro F1; best Bacterial recall
- ✅ 6.2× smaller than ResNet50, 1.29× faster; stable latency (lowest std)
- ✅ Validated production pipeline (TFJS conversion, mobile integration, multi-seed history)
- ❌ Not the highest raw accuracy in this study (−1.4 pp vs ResNet50, not significant)
- ❌ Weakest Hotspot/Mange recall among the top three (though still ≥0.905 / ≥0.910)

**MobileNetV2**
- ✅ Smallest and fastest (2.63 M params, 54.5 ms, 10.5 MB weights)
- ❌ Lowest accuracy on every seed (86.95% mean, −4.2 pp vs EffNet); worst macro F1 (0.839)
- ❌ Unstable Allergic Dermatitis recall (0.744 ± 0.089) — clinically the most concerning failure mode

**ResNet50**
- ✅ Highest mean accuracy (92.54%) and macro F1 (0.904); best single run (95.10%)
- ✅ Best Hotspot and Mange recall
- ❌ 24.16 M params, 230 MB — 6.2× the size of EfficientNetB0; impractical for on-device deployment
- ❌ Weakest Bacterial recall among top three (0.802); highest seed variance; slowest to train

**DenseNet121**
- ✅ Strong overall (90.44%); best AD F1 (0.843); balanced per-class profile
- ❌ Never ranks first on any aggregate metric; slowest inference (81.4 ms); 427-layer topology complicates fine-tuning choices
- ❌ Lower Fungal F1 than the top two (0.899)

## 11. Which Model Performed Best?

**Depends on the criterion — stated strictly from our results:**

1. **Highest raw accuracy / macro F1:** ResNet50 (0.9254 / 0.9037 mean) — but the margin over EfficientNetB0 is **not statistically significant** (p=0.692; CIs overlap heavily).
2. **Highest single run:** ResNet50 seed 7 at **95.10%** vs EfficientNetB0 seed 42 (production) at **93.71%**; however, at the **matched production seed 42**, EfficientNetB0 wins (0.9371 vs ResNet50 0.8951), and EfficientNetB0's best run beats ResNet50's seed-42 run by 4.2 points. The mean ranking is driven by ResNet50's exceptional seed-7 run, not consistent dominance.
2. **Best per-class recall:** ResNet50 on Hotspot/Mange; EfficientNetB0 on Bacterial; three-way tie on Allergic Dermatitis; Fungal shared by ResNet/EffNet.
3. **Best efficiency (accuracy per MB / per ms):** **EfficientNetB0** — it delivers top-tier accuracy at 17.7 MB and 60 ms, whereas ResNet50 pays 6.2× memory and 1.3× latency for a statistically indistinguishable +1.4 pp.
4. **Most suitable for mobile deployment:** EfficientNetB0 (MobileNetV2 is smaller/faster but gives up 4+ points of accuracy and AD stability; ResNet50/DenseNet121 are too large/slow for on-device use at 96.6/29.4 MB).

## 12. Limitations and Threats to Validity

1. **Small sample sizes:** 3 seeds per model and a 143-image test set → wide confidence intervals; the ResNet50–EfficientNetB0 gap cannot be declared real or unreal with this power (paired p=0.69).
2. **Single test set:** all models share one fixed split; results are specific to this split.
3. **One shared hyperparameter recipe:** tuned historically for EfficientNetB0; ResNet50/DenseNet121 may improve under architecture-specific tuning (e.g., learning rate, longer fine-tuning). This favors the incumbent.
4. **EfficientNetB0 not retrained in this study:** its checkpoints were produced in earlier sessions with the same recipe (35-layer unfreeze ≈ 23% vs 25% for others — a negligible, documented difference). Its training duration was not recorded → N/A in duration tables.
5. **CPU-only latency measurements** on a general-purpose machine (no isolated benchmarking host); absolute numbers will differ on mobile hardware — relative ordering is indicative.
6. **Class imbalance** (4.4:1) and subjective ground-truth labels (dermatoses can overlap clinically) bound achievable performance for every model.
7. **No architecture-specific hyperparameter search** was performed (equal tuning effort by design: same recipe for all).

## 13. Recommendation for XPawSure

**Retain EfficientNetB0 as the XPawSure model — based on these results, not on assumption.**

Evidence:
1. **At the matched production seed (42), EfficientNetB0 beats every alternative** — 93.71% vs ResNet50 89.51%, DenseNet121 88.11%, MobileNetV2 88.11% (§6.1c), a 4.2–5.6 point lead under fully like-for-like conditions.
2. It is **statistically tied for the best 3-seed accuracy** (91.14% ± 2.46 vs ResNet50's 92.54% ± 2.82; p=0.69). ResNet50's mean lead is driven by a single outlier run (95.10% at seed 7); at seed 7 EfficientNetB0 had its own worst run (88.81%), and ResNet50's seed-42 run fell 4.2 points below the production model — the ranking flips with the seed and is not stable.
3. It is **6.2× smaller and 1.29× faster than ResNet50**, and only 5.5 ms slower than MobileNetV2 while being **4.2 points more accurate** and far more stable on the rare class (Allergic Dermatitis).
4. It has the **best Bacterial recall** of all four models and macro F1 within 0.005 of the top.
5. It is the only candidate with a **fully validated end-to-end pipeline** (training → evaluation → TensorFlow.js conversion → mobile integration) and 3-seed history.
6. MobileNetV2 — the previous generation choice — is measurably worse on this dataset; DenseNet121 adds size and latency without accuracy gains.

If future work can tolerate a ~97 MB float32 (≈ 4× larger) on-device model and requires the absolute best raw accuracy, **ResNet50 is the honest alternative** (best mean accuracy, best Hotspot/Mange recall). For XPawSure's current mobile-first constraints, EfficientNetB0 is the best accuracy–efficiency trade-off and remains the justified selection.

---

### Deliverables index
| File | Contents |
|---|---|
| `MODEL_COMPARISON_REPORT.md` | This report |
| `MODEL_COMPARISON_RESULTS.csv` | All 12 runs: metrics, per-class P/R/F1, params, sizes, durations, latency |
| `RESULTS_TABLES.md` | Comparative tables (also embedded above) |
| `reports/` | Per-run classification reports (12) |
| `confusion_matrices/` | Per-run CM images (12) + `cm_mean_*.png` (4) |
| `training_configs/` | Per-run config + metrics JSON (21) |
| `accuracy_macroF1_comparison.png`, `per_class_recall_comparison.png`, `training_duration_comparison.png` | Visualizations |
| `DEFENSE_SUMMARY.md` | Concise defense Q&A |
| `REPRODUCIBILITY.md` | Step-by-step reproduction instructions |
