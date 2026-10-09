import argparse
from ml.normalization import compute_shared_scales

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--out",default="output/target_scales.json")
    ap.add_argument("--h-dry",type=float,default=1e-4); ap.add_argument("--stride",type=int,default=2); ap.add_argument("--max-runs",type=int,default=None)
    a=ap.parse_args(); print(compute_shared_scales(a.manifest,a.out,a.h_dry,a.stride,a.max_runs))
