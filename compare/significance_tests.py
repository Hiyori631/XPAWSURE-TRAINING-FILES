import json, glob
import numpy as np

runs = {}
for p in sorted(glob.glob('compare/results/*_metrics.json')):
    m = json.load(open(p))
    runs.setdefault(m['label'], {})[m['seed']] = m

# paired t-test ResNet50 vs EfficientNetB0 (by seed)
def paired(a, b):
    diffs = np.array([runs[a][s]['test_accuracy'] - runs[b][s]['test_accuracy'] for s in (42, 7, 123)])
    mean, sd = diffs.mean(), diffs.std(ddof=1)
    t = mean / (sd / np.sqrt(3))
    return diffs, mean, sd, t

for other in ['MobileNetV2', 'ResNet50', 'DenseNet121']:
    diffs, mean, sd, t = paired('EfficientNetB0', other)
    print(f'EffNet - {other}: diffs={np.round(diffs,4)} mean={mean:+.4f} sd={sd:.4f} t={t:.3f} (df=2, two-tail p~see below)')

# t-distribution p-value for df=2 (no scipy needed? scipy available)
from scipy import stats
for other in ['MobileNetV2', 'ResNet50', 'DenseNet121']:
    diffs, mean, sd, t = paired('EfficientNetB0', other)
    p = 2 * (1 - stats.t.cdf(abs(t), df=2))
    ci = stats.t.ppf(0.975, df=2) * sd / np.sqrt(3)
    print(f'EffNet vs {other}: mean diff {mean:+.4f}, 95% CI [{mean-ci:+.4f}, {mean+ci:+.4f}], p={p:.3f}')

# 95% CI for each model mean
print('\n95% CI of mean accuracy (df=2, t=4.303):')
for label, seeds in runs.items():
    v = np.array([seeds[s]['test_accuracy'] for s in (42, 7, 123)])
    ci = 4.3027 * v.std(ddof=1) / np.sqrt(3)
    print(f'  {label:16s}: {v.mean():.4f} ± {ci:.4f}  [{v.mean()-ci:.4f}, {v.mean()+ci:.4f}]')

print('\n95% CI of mean macro F1:')
for label, seeds in runs.items():
    v = np.array([seeds[s]['macro_f1'] for s in (42, 7, 123)])
    ci = 4.3027 * v.std(ddof=1) / np.sqrt(3)
    print(f'  {label:16s}: {v.mean():.4f} ± {ci:.4f}  [{v.mean()-ci:.4f}, {v.mean()+ci:.4f}]')
