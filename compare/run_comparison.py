"""Sequential orchestrator for the model comparison study.

Runs: 3 alternative backbones x 3 seeds x 2 stages (CPU, one at a time).
Resumable: skips stages whose output artifacts already exist.
EfficientNetB0 baseline uses existing production checkpoints (evaluated
separately by evaluate_backbone.py) and is NOT retrained here.

Usage (from repo root):
  python compare/run_comparison.py
"""
import os, sys, subprocess, time, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--backbones", type=str, default="mobilenet,resnet50,densenet")
parser.add_argument("--seeds", type=str, default="42,7,123")
parser.add_argument("--epochs", type=int, default=None,
                    help="Override epochs (smoke/testing only)")
args = parser.parse_args()

BACKBONES = [b.strip() for b in args.backbones.split(",") if b.strip()]
SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]

CKPT_DIR = os.path.join(ROOT, "compare", "ckpt")
RES_DIR = os.path.join(ROOT, "compare", "results")
LOG_DIR = os.path.join(ROOT, "compare", "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

STAGE_SCRIPT = os.path.join(ROOT, "compare", "train_backbone.py")


def run(cmd, log_path):
    print(f"\n>>> {time.strftime('%H:%M:%S')} RUNNING: {' '.join(cmd)}")
    with open(log_path, "w") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=ROOT,
        )
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FAILED (exit {proc.returncode}): {' '.join(cmd)}\nSee {log_path}")
    print(f"<<< OK {os.path.basename(log_path)}")


def done(bb, seed, stage):
    if stage == "frozen":
        return os.path.exists(os.path.join(CKPT_DIR, f"{bb}_s{seed}_frozen_best.keras"))
    return os.path.exists(os.path.join(RES_DIR, f"{bb}_s{seed}_metrics.json"))


t_start = time.time()
plan = []
for bb in BACKBONES:
    for seed in SEEDS:
        for stage in ("frozen", "finetune"):
            plan.append((bb, seed, stage))

print(f"Plan: {len(plan)} stages | backbones={BACKBONES} seeds={SEEDS}")
for i, (bb, seed, stage) in enumerate(plan, 1):
    if done(bb, seed, stage):
        print(f"[{i}/{len(plan)}] SKIP {bb} s{seed} {stage} (already done)")
        continue
    cmd = [sys.executable, STAGE_SCRIPT, "--backbone", bb, "--seed", str(seed), "--stage", stage]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    log = os.path.join(LOG_DIR, f"{bb}_s{seed}_{stage}.log")
    for attempt in (1, 2):
        try:
            run(cmd, log)
            break
        except RuntimeError as e:
            print(f"ERROR attempt {attempt}: {e}")
            if attempt == 2:
                sys.exit(1)
            time.sleep(5)

elapsed = time.time() - t_start
print(f"\nAll stages complete in {elapsed/3600:.2f} h ({elapsed/60:.0f} min).")
with open(os.path.join(RES_DIR, "run_comparison_summary.json"), "w") as f:
    json.dump({"elapsed_s": round(elapsed, 1), "backbones": BACKBONES, "seeds": SEEDS}, f, indent=2)
