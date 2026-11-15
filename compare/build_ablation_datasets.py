"""Build the 3-arm data-source ablation datasets (Local / Public / Combined).

Reads existing datasets READ-ONLY; creates only AblationDatasets/.
Method:
  1. Pool Local + Public images, dedup by MD5 then decoded-pixel near-dup
     (Pearson r > 0.95 => same photo, different encoding).
  2. Stratified 15%/class global holdout (seed 42) = CommonTest slices,
     excluded from every arm's training data.
  3. Remaining pool split per domain; each arm gets 85/15 train/valid (seed 42).
  4. Arm Test dir = that arm's domain slice of the global holdout.
Classes: Bacterial, Fungal, Hotspot, Mange (4-class; AD not in local/public).
"""
import os, shutil, hashlib, random, json
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

LOCAL_SRC = "ProcessedDataset224_Local"
PUBLIC_SRC = "ProcessedDataset224"
OUT = "AblationDatasets"
CLASSES = ["Bacterial", "Fungal", "Hotspot", "Mange"]
SEED = 42
HOLDOUT = 0.15
VALID_FRAC = 0.15  # of remaining dev pool
NEAR_DUP_R = 0.95

random.seed(SEED)
np.random.seed(SEED)


def list_images(d):
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif")):
            out.append(os.path.join(d, f))
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def pixels(path, size=(64, 64)):
    return np.asarray(Image.open(path).convert("L").resize(size), dtype=np.float64).flatten()


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else 0.0


# ---------- 1. collect sources ----------
records = []  # dict(domain, cls, path, md5)
for domain, src in [("local", LOCAL_SRC), ("public", PUBLIC_SRC)]:
    for cls in CLASSES:
        for p in list_images(os.path.join(src, "Train", cls)) + \
                 list_images(os.path.join(src, "Valid", cls)) + \
                 list_images(os.path.join(src, "Test", cls)):
            records.append({"domain": domain, "cls": cls, "path": p})
        # Public uses capitalized splits; Local too. Both checked.
print(f"Collected {len(records)} source files (local={sum(r['domain']=='local' for r in records)}, "
      f"public={sum(r['domain']=='public' for r in records)})")

# ---------- 2. MD5 dedup (keep first: local wins over public for same bytes) ----------
seen_md5 = {}
md5_dupes = 0
uniq = []
for r in records:
    r["md5"] = md5(r["path"])
    if r["md5"] in seen_md5:
        md5_dupes += 1
        continue
    seen_md5[r["md5"]] = r
    uniq.append(r)
print(f"MD5 dedup: removed {md5_dupes} byte-identical -> {len(uniq)} unique files")

# ---------- 3. near-dup dedup (decoded pixels; local wins) ----------
uniq.sort(key=lambda r: (r["cls"], 0 if r["domain"] == "local" else 1, r["path"]))
kept = []
by_class = defaultdict(list)
near_dupes = 0
for r in uniq:
    r["pix"] = pixels(r["path"])
    dup = False
    for k in by_class[r["cls"]]:
        if pearson(r["pix"], k["pix"]) > NEAR_DUP_R:
            dup = True
            break
    if dup:
        near_dupes += 1
        continue
    by_class[r["cls"]].append(r)
    kept.append(r)
for r in kept:
    del r["pix"]
print(f"Near-dup dedup (r>{NEAR_DUP_R}): removed {near_dupes} re-encoded copies -> {len(kept)} unique photos")
for cls in CLASSES:
    c = [r for r in kept if r["cls"] == cls]
    print(f"  {cls}: {len(c)} (local={sum(r['domain']=='local' for r in c)}, "
          f"public={sum(r['domain']=='public' for r in c)})")

# ---------- 4. global holdout 15%/class ----------
holdout, dev = [], []
for cls in CLASSES:
    items = [r for r in kept if r["cls"] == cls]
    random.shuffle(items)
    n_test = max(1, int(round(len(items) * HOLDOUT)))
    holdout.extend(items[:n_test])
    dev.extend(items[n_test:])
print(f"\nGlobal holdout (common test): {len(holdout)} images")
for cls in CLASSES:
    h = [r for r in holdout if r["cls"] == cls]
    print(f"  {cls}: {len(h)} (local={sum(r['domain']=='local' for r in h)}, "
          f"public={sum(r['domain']=='public' for r in h)})")

# ---------- 5. per-arm dev split 85/15 train/valid ----------
def split_dev(items):
    train, valid = [], []
    for cls in CLASSES:
        c = [r for r in items if r["cls"] == cls]
        random.shuffle(c)
        n_val = int(round(len(c) * VALID_FRAC))
        valid.extend(c[:n_val])
        train.extend(c[n_val:])
    return train, valid


arms = {
    "Local": [r for r in dev if r["domain"] == "local"],
    "Public": [r for r in dev if r["domain"] == "public"],
    "Combined": list(dev),
}
# Test slices
test_slices = {
    "Local": [r for r in holdout if r["domain"] == "local"],
    "Public": [r for r in holdout if r["domain"] == "public"],
    "Combined": list(holdout),
}

# ---------- 6. write ----------
if os.path.isdir(OUT):
    print(f"\nWARNING: {OUT} exists — removing and rebuilding")
    shutil.rmtree(OUT)

summary = {"holdout": len(holdout), "arms": {}}
for arm, items in arms.items():
    train, valid = split_dev(items)
    tests = test_slices[arm]
    for split, recs in [("Train", train), ("Valid", valid), ("Test", tests)]:
        for r in recs:
            dst_dir = os.path.join(OUT, arm, split, r["cls"])
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(r["path"], os.path.join(dst_dir, os.path.basename(r["path"])))
    summary["arms"][arm] = {
        "train": len(train), "valid": len(valid), "test": len(tests),
        "test_local": sum(r["domain"] == "local" for r in tests),
        "test_public": sum(r["domain"] == "public" for r in tests),
    }
    print(f"{arm:9s} Train={len(train):4d} Valid={len(valid):3d} Test={len(tests):3d} "
          f"(test: local={summary['arms'][arm]['test_local']}, public={summary['arms'][arm]['test_public']})")

with open(os.path.join(OUT, "_provenance.json"), "w") as f:
    json.dump({
        "sources": {LOCAL_SRC: "local", PUBLIC_SRC: "public/international"},
        "source_files": len(records),
        "md5_dupes_removed": md5_dupes,
        "near_dupes_removed": near_dupes,
        "unique_photos": len(kept),
        "holdout_fraction": HOLDOUT,
        "seed": SEED,
        **summary,
    }, f, indent=2)

print(f"\nDone. Dataset tree written to {OUT}/ (existing datasets untouched).")
