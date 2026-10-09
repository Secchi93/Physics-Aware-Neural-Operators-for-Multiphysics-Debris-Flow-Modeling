"""Smoke tests for a clean checkout; no large downloads required for static checks."""
import argparse
import ast
import subprocess
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",action="store_true",help="Require Zenodo dataset and scales")
    ap.add_argument("--solver",action="store_true",help="Require the external PyDebrisFlow2D installation")
    ap.add_argument("--gpu",action="store_true",help="Run a one-batch smoke test for all six models")
    a=ap.parse_args()
    for path in ROOT.rglob("*.py"):
        if "__pycache__" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    print("[PASS] Python syntax")
    files=list((ROOT/"configs_ml/comparison/deep200").glob("*.yaml"))
    assert len(files)==6, f"Expected six benchmark configs, found {len(files)}"
    for config in files:
        obj=yaml.safe_load(config.read_text())
        assert obj["data"]["manifest"]=="output/dataset/manifest.csv"
        assert obj["data"]["target_scales"]=="output/target_scales.json"
    print("[PASS] Six YAML paths and configurations")
    subprocess.run([sys.executable,str(ROOT/"validate_protocol.py"),"--tier","deep200"],cwd=ROOT,check=True)
    if a.solver:
        import pydebrisflow
        print("[PASS] PyDebrisFlow2D:",pydebrisflow.__file__)
    if a.data or a.gpu:
        assert (ROOT/"output/dataset/manifest.csv").exists(),"Missing Zenodo dataset manifest"
        assert (ROOT/"output/target_scales.json").exists(),"Missing Zenodo normalization scales"
        print("[PASS] Data and scales found")
    if a.gpu:
        subprocess.run([sys.executable,str(ROOT/"run.py"),"--stage","smoke"],cwd=ROOT,check=True)
    print("[PASS] All requested smoke checks completed")
if __name__=="__main__":main()
