#!/usr/bin/env python3
"""
invert_mrc.py
-------------
Utility script to invert/flip density contrast in MRC 3D volumes (e.g. for ChimeraX visualization).

Usage:
  python invert_mrc.py <input_mrc_path> [--output_dir <dir>] [--output_mrc <path>] [--mode <mode>]
"""

import argparse
from pathlib import Path
import mrcfile
import numpy as np


def invert_mrc(input_path: Path, output_path: Path, mode: str = "negate"):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with mrcfile.open(str(input_path), permissive=True) as mrc:
        data = mrc.data.copy().astype(np.float32)
        voxel_size = mrc.voxel_size

    if mode == "negate":
        # Standard contrast flip: protein becomes positive (> 0)
        inverted = -data
    elif mode == "min_max_flip":
        inverted = (data.max() + data.min()) - data
    elif mode == "normalize_invert":
        vmin, vmax = float(data.min()), float(data.max())
        norm = (data - vmin) / (vmax - vmin + 1e-8)
        inverted = 1.0 - norm
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with mrcfile.new(str(output_path), overwrite=True) as mrc_out:
        mrc_out.set_data(inverted.astype(np.float32))
        mrc_out.voxel_size = voxel_size

    print(f"✓ Successfully saved inverted MRC to: {output_path}")
    print(f"  Original: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    print(f"  Inverted: min={inverted.min():.4f}, max={inverted.max():.4f}, mean={inverted.mean():.4f}")


def main():
    p = argparse.ArgumentParser(description="Invert MRC density values for ChimeraX visualization")
    p.add_argument("input_mrc", help="Path to input .mrc file")
    p.add_argument("--output_dir", default=None, help="Directory to save inverted MRC (default: temp/ or same as input)")
    p.add_argument("--output_mrc", default=None, help="Explicit output file path")
    p.add_argument("--mode", default="negate", choices=["negate", "min_max_flip", "normalize_invert"],
                   help="Inversion mode (default: negate)")

    args = p.parse_args()
    in_path = Path(args.input_mrc)

    if args.output_mrc:
        out_path = Path(args.output_mrc)
    elif args.output_dir:
        out_path = Path(args.output_dir) / f"{in_path.stem}_inverted.mrc"
    else:
        out_path = Path("temp") / f"{in_path.stem}_inverted.mrc"

    invert_mrc(in_path, out_path, mode=args.mode)


if __name__ == "__main__":
    main()
