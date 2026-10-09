"""Six-model benchmark execution; run from the repository root."""
import argparse
import subprocess
import sys
from pathlib import Path

CASES = ("fno_cartesian", "gno_cartesian", "gino_cartesian", "fno_adaptive", "gno_adaptive", "gino_adaptive")
TIER = "deep200"

def run(*args):
    command = [sys.executable, *map(str,args)]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validate","smoke","train","evaluate","all"], default="validate")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    a=ap.parse_args()
    run("validate_protocol.py", "--tier", TIER)
    if a.stage == "validate": return
    for case in CASES:
        config=Path("configs_ml/comparison")/TIER/(case+".yaml")
        if a.stage in ("smoke",):
            run("train.py", "--config",config,"--seed",a.seed,"--smoke-test","--profile-first-batch")
        if a.stage in ("train","all"):
            cmd=["train.py","--config",config,"--seed",a.seed]
            if a.overwrite: cmd.append("--overwrite-run")
            run(*cmd)
        if a.stage in ("evaluate","all"):
            checkpoint=Path("output/models")/TIER/(case+"_seed"+str(a.seed))/"best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}; download models from Zenodo or train the model")
            run("evaluation.py","--checkpoint",checkpoint,"--splits","validation","iid_test","single_ood_test","joint_ood_test")
if __name__ == "__main__":main()
