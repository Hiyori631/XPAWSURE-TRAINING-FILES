"""Sequential orchestrator for the data-source ablation (EfficientNetB0 only).

Runs: 3 arms (local/public/combined) x 3 seeds x 2 stages (CPU, one at a time).
Resumable: skips stages whose output artifacts already exist.

Usage (from repo root):
  python compare/run_ablation.py
"""
import os, sys, subprocess, time, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--arms", type=str, default="Local,Public,Combined")
parser.add_argument("--seeds", type=str, default="42,7,123")
parser.add_argument("--epochs", type=int, default=None,
                    help="Override epochs (smoke/testing only)")
args = parser.parse_args()

ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]
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


def done(arm, seed, stage):
    prefix = f"{arm.lower()}_effnet_s{seed}"
    if stage == "frozen":
        return os.path.exists(os.path.join(CKPT_DIR, f"{prefix}_frozen_best.keras"))
    return os.path.exists(os.path.join(RES_DIR, f"{prefix}_metrics.json"))


t_start = time.time()
plan = []
for arm in ARMS:
    for seed in SEEDS:
        for stage in ("frozen", "finetune"):
            plan.append((arm, seed, stage))

print(f"Plan: {len(plan)} stages | arms={ARMS} seeds={SEEDS}")
for i, (arm, seed, stage) in enumerate(plan, 1):
    if done(arm, seed, stage):
        print(f"[{i}/{len(plan)}] SKIP {arm} s{seed} {stage} (already done)")
        continue
    cmd = [
        sys.executable, STAGE_SCRIPT,
        "--backbone", "effnet",
        "--seed", str(seed),
        "--stage", stage,
        "--dataset", f"AblationDatasets/{arm}",
        "--tag_prefix", f"{arm.lower()}_",
    ]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    log = os.path.join(LOG_DIR, f"ablation_{arm.lower()}_s{seed}_{stage}.log")
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
with open(os.path.join(RES_DIR, "run_ablation_summary.json"), "w") as f:
    json.dump({"elapsed_s": round(elapsed, 1), "arms": ARMS, "seeds": SEEDS}, f, indent=2)
