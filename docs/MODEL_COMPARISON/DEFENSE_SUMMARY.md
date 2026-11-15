# XPawSure Model Comparison — Defense Summary

*Quick-reference answers for capstone defense. All numbers from the 3-seed study in `MODEL_COMPARISON_REPORT.md`.*

---

**Q: Why did we choose EfficientNetB0?**
Because it offers the best accuracy–efficiency balance: **at the matched production seed (42) it beats every alternative — 93.71% vs ResNet50 89.51%, DenseNet121 88.11%, MobileNetV2 88.11%** — and over 3 seeds it is statistically tied with the highest-accuracy model (91.14% vs 92.54%, paired p=0.69) while being 6.2× smaller (17.7 MB, ~60 ms) than ResNet50 (96.6 MB, 77 ms). It also already runs end-to-end in the XPawSure mobile pipeline.

**Q: How does it compare with MobileNetV2?**
EfficientNetB0 is **+4.2 points more accurate** (91.14% vs 86.95%) and won all three seed-pair comparisons. MobileNetV2 is only 5.5 ms faster and 7 MB smaller, but it is the weakest model overall: lowest macro F1 (0.839) and unstable Allergic Dermatitis recall (0.744 ± 0.089 vs 0.923 ± 0.000). The size/speed saving is not worth the accuracy and clinical-stability loss.

**Q: How does it compare with ResNet50?**
ResNet50 achieved the **highest raw mean accuracy (92.54%) and macro F1 (0.904)** — 1.4 points above EfficientNetB0 — but the difference is **not statistically significant** (paired t-test p=0.692; 95% CIs overlap heavily). In exchange, ResNet50 is **6.2× larger (96.6 MB vs 17.7 MB weights)**, 1.29× slower, and had the weakest Bacterial recall of the top three (0.802 vs 0.864). EfficientNetB0 wins on efficiency; ResNet50 wins on raw accuracy — honestly reported.

**Q: How does it compare with DenseNet121?**
EfficientNetB0 is +0.7 points more accurate (91.14% vs 90.44%, p=0.828), has better Fungal F1 (0.950 vs 0.899) and Bacterial recall (0.864 vs 0.803), and is 1.6× faster (60 vs 81.4 ms) with 40% fewer parameters. DenseNet121's only clear win is Allergic Dermatitis F1 (0.843 vs 0.821).

**Q: Which model has the highest accuracy?**
**ResNet50** on 3-seed mean — 92.54% ± 2.82 (best single run: 95.10%, seed 7). EfficientNetB0's **production model (seed 42) is 93.71%** (134/143), which beats ResNet50's seed-42 run (89.51%) by 4.2 points; EfficientNetB0's 3-seed mean is 91.14% ± 2.46 (seeds 93.71 / 88.81 / 90.91). The mean gap (1.4 pp) is within noise (p=0.69).

**Q: Which model has the best macro F1-score?**
**ResNet50** — 0.9037 ± 0.0295, followed by EfficientNetB0 0.8988 ± 0.0302 (difference 0.005, not significant).

**Q: Which model has the best per-class recall?**
Mixed: **ResNet50** best on Hotspot (0.952, perfect 20/21 in all seeds) and Mange (0.962); **EfficientNetB0** best on Bacterial (0.864); Allergic Dermatitis tied at 0.923 for EfficientNetB0/ResNet50/DenseNet121; Fungal tied EfficientNetB0/ResNet50 (0.956).

**Q: Which model is most suitable for mobile deployment?**
**EfficientNetB0.** MobileNetV2 is marginally faster (54.5 vs 60.0 ms) and smaller (10.5 vs 17.7 MB) but costs 4.2 accuracy points and AD stability. ResNet50 (96.6 MB) and DenseNet121 (29.4 MB, 81.4 ms) are too heavy/slow for on-device inference. EfficientNetB0 is the only model that combines top-tier accuracy with a validated TensorFlow.js mobile pipeline.

**Q: So why is EfficientNetB0 the most appropriate choice?**
Because the experiments show it occupies the Pareto-optimal point for XPawSure's requirements: statistically tied for best accuracy, second-best macro F1, best Bacterial recall, ~17.7 MB, ~60 ms, and a proven deployment path. Choosing ResNet50 would buy a non-significant +1.4 points at 6.2× the model size — unacceptable for a mobile app. Choosing MobileNetV2 would save 5.5 ms at a significant real cost in accuracy and rare-class stability. EfficientNetB0 is justified **by these results**, with the honest caveat that ResNet50 holds the raw-accuracy crown in this study.

---

### Key numbers at a glance (143-image test set)

| Model | Accuracy (mean ± std, 3 seeds) | Seed 42 (matched) | Macro F1 (mean) | Weights | Latency | Rank |
|---|---|---|---|---|---|---|
| EfficientNetB0 | 0.9114 ± 0.0246 | **0.9371** ← production | 0.8988 ± 0.0302 | 17.7 MB | 60.0 ms | **Best trade-off (chosen)** |
| ResNet50 | **0.9254 ± 0.0282** | 0.8951 | **0.9037 ± 0.0295** | 96.6 MB | 77.4 ms | Best raw mean, 6.2× larger |
| DenseNet121 | 0.9044 ± 0.0246 | 0.8811 | 0.8897 ± 0.0256 | 29.4 MB | 81.4 ms | Balanced, no clear win |
| MobileNetV2 | 0.8695 ± 0.0201 | 0.8811 | 0.8391 ± 0.0186 | 10.5 MB | 54.5 ms | Smallest/fastest, weakest |

*At the matched production seed 42, EfficientNetB0 leads all alternatives by 4.2–5.6 points. Significance: no pairwise 3-seed difference reached p < 0.05 (EffNet vs ResNet p=0.69; vs DenseNet p=0.83; vs MobileNet p=0.14).*
