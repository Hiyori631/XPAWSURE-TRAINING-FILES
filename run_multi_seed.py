import os, sys, subprocess, json, time

PIPELINES = {
    "mobilenetv2": {"train": "train.py", "fine_tune": "fine_tune.py", "seeds": [42, 7, 123, 999, 777]},
    "effnet": {"train": "train_effnet.py", "fine_tune": "fine_tune_effnet.py", "seeds": [42, 7, 123]},
    "effnet_v2": {"train": "train_effnet.py", "fine_tune": "fine_tune_effnet_two_stage.py", "seeds": [42, 7, 123]},
    "effnet_cutmix": {"train": "train_effnet.py", "fine_tune": "fine_tune_effnet_cutmix.py", "seeds": [42, 7]},
}

def run_phase(script, seed, run_name, dataset, timeout=14400):
    print(f"\n>>> [{time.strftime('%H:%M:%S')}] Running: {script} --seed {seed} --run_name {run_name} --dataset {dataset}")
    result = subprocess.run(
        [sys.executable, script, "--seed", str(seed), "--run_name", run_name, "--dataset", dataset],
        capture_output=True, text=True, timeout=timeout
    )
    for line in result.stdout.splitlines():
        print(line)
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"STDERR: {line}")
    return result

def parse_acc(output):
    for line in output.split('\n'):
        if 'Test Accuracy:' in line:
            try:
                return float(line.split(':')[1].strip())
            except:
                pass
    return None

def main():
    import argparse as ap
    parser = ap.ArgumentParser(description="Multi-seed training runner")
    parser.add_argument("--pipeline", choices=list(PIPELINES.keys()), default="mobilenetv2")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--dataset", type=str, default="ProcessedDataset224_PublicLocal")
    args = parser.parse_args()

    cfg = PIPELINES[args.pipeline]
    seeds = args.seeds if args.seeds else cfg["seeds"]
    os.makedirs("models", exist_ok=True)

    ds_tag = ""
    if args.dataset != "ProcessedDataset224_PublicLocal":
        ds_tag = os.path.basename(args.dataset).replace("ProcessedDataset224_", "").lower() + "_"

    results = []
    for seed in seeds:
        run_name = f"{ds_tag}s{seed}"
        print(f"\n{'='*60}")
        print(f"PIPELINE: {args.pipeline} | SEED: {seed} | RUN: {run_name}")
        print(f"{'='*60}")

        frozen_acc = None
        ft_acc = None

        r1 = run_phase(cfg["train"], seed, run_name, args.dataset, timeout=args.timeout)
        if r1.returncode == 0:
            frozen_acc = parse_acc(r1.stdout)
        else:
            print(f"FAILED: {cfg['train']} for seed {seed}")
            continue

        r2 = run_phase(cfg["fine_tune"], seed, run_name, args.dataset, timeout=args.timeout)
        if r2.returncode == 0:
            ft_acc = parse_acc(r2.stdout)
        else:
            print(f"FAILED: {cfg['fine_tune']} for seed {seed}")

        results.append({"seed": seed, "frozen_acc": frozen_acc, "fine_tune_acc": ft_acc})
        print(f"\n>>> Seed {seed}: Frozen={frozen_acc:.4f}, Fine-tune={ft_acc}" + (f" ({ft_acc:.4f})" if ft_acc else " (N/A)"))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        ft = f"{r['fine_tune_acc']:.4f}" if r['fine_tune_acc'] else "N/A"
        fr = f"{r['frozen_acc']:.4f}" if r['frozen_acc'] else "N/A"
        print(f"  Seed {r['seed']:>3}: Frozen={fr}, Fine-tune={ft}")

    valid = [r for r in results if r['fine_tune_acc'] is not None]
    if valid:
        best = max(valid, key=lambda r: r['fine_tune_acc'])
        print(f"\nBEST: Seed {best['seed']} with {best['fine_tune_acc']:.4f}")

    report = {"pipeline": args.pipeline, "seeds": seeds, "results": results, "best": best if valid else None}
    report_path = f"models/{args.pipeline}_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {report_path}")

if __name__ == "__main__":
    main()
