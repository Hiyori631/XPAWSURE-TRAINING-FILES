# Reproducibility Instructions — Model Comparison Study

## Environment
- Python 3.x with `venv/` (TensorFlow **2.21.0**, Keras 3.15, scikit-learn 1.9, matplotlib 3.11, numpy 2.4)
- CPU-only (native Windows TF ≥ 2.11 has no GPU support) — expect **~10–15 h** total for 9 alternative runs
- ImageNet weights for MobileNetV2/EfficientNetB0/ResNet50/DenseNet121 are downloaded automatically on first build (~100 MB total, cached in `~/.keras/models`)

## Dataset
`ProcessedDataset224_Clean_5class/` (641/136/143, 5 classes) — **must already exist and remain unmodified**.
All scripts read it read-only; never point `--dataset` at anything else for this study.

## Directory layout produced by this study
```
compare/
  train_backbone.py        # backbone-agnostic trainer (--backbone --seed --stage [--epochs --max_batches])
  evaluate_backbone.py     # unified evaluator for any .keras checkpoint
  run_comparison.py        # sequential orchestrator (resumable, skips completed stages)
  aggregate_results.py     # builds CSV + tables + charts in docs/MODEL_COMPARISON/
  benchmark_latency.py     # warm-up-excluded CPU latency benchmark (updates metrics JSONs)
  ckpt/                    # 12 alternative-model checkpoints
  results/                 # metrics JSONs, logs, reports, CMs, configs, orchestrator logs
docs/MODEL_COMPARISON/     # final deliverables
```

## Reproduce from scratch
```powershell
# 1. Train the 3 alternative backbones × seeds {42,7,123} × 2 stages (resumable)
python compare\run_comparison.py
#    Skips any stage whose output already exists. To force a redo, delete the
#    corresponding files in compare\ckpt\ and compare\results\.

# 2. Re-evaluate the EfficientNetB0 production baseline (read-only on models/)
python compare\evaluate_backbone.py --model models\fine_tuned_best_effnet_clean_5class_s42.keras  --tag effnet_s42  --label EfficientNetB0
python compare\evaluate_backbone.py --model models\fine_tuned_best_effnet_clean_5class_s7.keras   --tag effnet_s7   --label EfficientNetB0
python compare\evaluate_backbone.py --model models\fine_tuned_best_effnet_clean_5class_s123.keras --tag effnet_s123 --label EfficientNetB0

# 3. Fair latency benchmark (warm-up excluded)
python compare\benchmark_latency.py

# 4. Aggregate → CSV, tables, charts
python compare\aggregate_results.py

# 5. Single-run example (any backbone)
python compare\train_backbone.py --backbone resnet50 --seed 42 --stage frozen
python compare\train_backbone.py --backbone resnet50 --seed 42 --stage finetune
#    Smoke test: add `--epochs 1 --max_batches 3`
```

## Protocol details (fixed for all models)
- Frozen: 100 ep, Adam + CosineDecay(1e-3→1e-5), SWA last 30%, early stop patience 15 (val_loss)
- Fine-tune: last ~25% of backbone layers, BN frozen, Adam 1e-5, 80 ep, SWA implicit via best-checkpoint, early stop patience 10
- EfficientNetB0 exception: production checkpoints use the original 35-layer unfreeze (≈23%) and were **not** retrained
- Loss: focal γ=2.0, class weights `{0:3.0, 1:2.0, 2:2.5, 3:1.0, 4:1.3}` (alphabetical class order)
- Test set evaluated only at the end of the fine-tune stage, once per run

## Verification of safety invariants
```powershell
# Nothing outside compare/ and docs/MODEL_COMPARISON/ should change:
Get-ChildItem models -Recurse -File | Where-Object LastWriteTime -gt (Get-Date).Date   # expect empty
(Get-ChildItem ProcessedDataset224_Clean_5class -Recurse -File).Count                  # expect 920
```
