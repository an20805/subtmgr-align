import argparse
import time
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
from aspire.utils.rotation import Rotation
from aspire.volume import Volume

from utils_BO import align_BO


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLUSTER_DIR = SCRIPT_ROOT / "data" / "0.1" / "Cluster_0"
DEFAULT_OUTPUT_DIR = SCRIPT_ROOT / "notebooks" / "cluster0_iterative_alignment_outputs"


def get_cluster_paths(cluster_dir: Path, n_expected: int | None = None):
    paths = sorted(cluster_dir.glob("*.mrc"), key=lambda p: int(p.stem))
    if n_expected is not None and len(paths) != n_expected:
        raise ValueError(f"Expected {n_expected} .mrc files, found {len(paths)} in {cluster_dir}")
    return paths


def load_mrc(path: Path, permissive: bool = False) -> np.ndarray:
    with mrcfile.open(path, permissive=permissive) as mrc:
        return np.array(mrc.data, dtype=np.float32)


def save_mrc(path: Path, arr: np.ndarray) -> Path:
    arr = np.asarray(arr, dtype=np.float32)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(arr)
    return path


def repair_mrc_headers(paths):
    repaired = []
    failures = []
    for path in paths:
        try:
            arr = load_mrc(path, permissive=True)
            save_mrc(path, arr)
            repaired.append(path.name)
        except Exception as exc:
            failures.append({"file_name": path.name, "error": str(exc)})
    return repaired, failures


def rotate_array_with_matrix(arr: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    vol = Volume(np.asarray(arr, dtype=np.float32))
    rotated = vol.rotate(Rotation(np.asarray(rotation_matrix, dtype=np.float32)))._data[0]
    return np.asarray(rotated, dtype=np.float32)


def compute_average(arrays) -> np.ndarray:
    total = None
    count = 0
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float32)
        if total is None:
            total = np.zeros_like(arr, dtype=np.float64)
        total += arr
        count += 1
    if count == 0:
        raise ValueError("Cannot compute an average from zero arrays.")
    return (total / count).astype(np.float32)


def flatten_rotation(rotation_matrix: np.ndarray) -> dict:
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)
    return {f"r{i}{j}": float(rotation_matrix[i, j]) for i in range(3) for j in range(3)}


def save_rotation_table(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def save_round_reference(output_dir: Path, round_idx: int, arr: np.ndarray) -> Path:
    path = output_dir / f"reference_round_{round_idx:02d}.mrc"
    save_mrc(path, arr)
    return path


def reference_stats(arr: np.ndarray, round_idx: int) -> dict:
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "round": round_idx,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "l2_norm": float(np.linalg.norm(arr)),
    }


def load_table_records(path: Path):
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path).to_dict("records")
    return []


def get_latest_full_reference_path(output_dir: Path):
    candidates = []
    for path in output_dir.glob("reference_round_*.mrc"):
        if path.stem.endswith("_partial"):
            continue
        if path.name == "final_reference.mrc":
            continue
        try:
            round_idx = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        candidates.append((round_idx, path))
    if not candidates:
        return None, 0
    latest_round, latest_path = max(candidates, key=lambda item: item[0])
    return latest_path, latest_round


def ensure_initial_reference(cluster_paths, output_dir: Path) -> tuple[np.ndarray, list[dict]]:
    initial_reference_path = output_dir / "reference_round_00.mrc"
    reference_stats_path = output_dir / "reference_stats.csv"

    if initial_reference_path.exists():
        initial_reference = load_mrc(initial_reference_path, permissive=False)
        if reference_stats_path.exists() and reference_stats_path.stat().st_size > 0:
            reference_stats_rows = pd.read_csv(reference_stats_path).to_dict("records")
        else:
            reference_stats_rows = [reference_stats(initial_reference, 0)]
            pd.DataFrame(reference_stats_rows).to_csv(reference_stats_path, index=False)
        return initial_reference, reference_stats_rows

    print("Computing initial reference from all subtomograms...")
    initial_reference = compute_average(load_mrc(path, permissive=False) for path in cluster_paths)
    save_round_reference(output_dir, 0, initial_reference)
    reference_stats_rows = [reference_stats(initial_reference, 0)]
    pd.DataFrame(reference_stats_rows).to_csv(reference_stats_path, index=False)
    return initial_reference, reference_stats_rows


def prepare_run_state(initial_reference_array: np.ndarray, output_dir: Path):
    metrics_rows = load_table_records(output_dir / "alignment_metrics.csv")
    final_rotations_rows = load_table_records(output_dir / "final_rotations.csv")

    reference_stats_path = output_dir / "reference_stats.csv"
    if reference_stats_path.exists() and reference_stats_path.stat().st_size > 0:
        reference_stats_rows = pd.read_csv(reference_stats_path).to_dict("records")
    else:
        reference_stats_rows = [reference_stats(initial_reference_array, 0)]

    latest_full_reference_path, latest_full_round = get_latest_full_reference_path(output_dir)
    if latest_full_reference_path is not None and latest_full_round >= 1:
        current_reference = load_mrc(latest_full_reference_path, permissive=False)
        start_round = latest_full_round + 1
        print(f"Resuming from completed round {latest_full_round}.")
    else:
        current_reference = initial_reference_array.copy()
        start_round = 1
        print("Starting from the initial reference.")

    return current_reference, start_round, metrics_rows, final_rotations_rows, reference_stats_rows


def align_round(reference_array: np.ndarray, cluster_paths, para, round_idx: int, verbose_every: int = 10):
    reference_volume = Volume(np.asarray(reference_array, dtype=np.float32))
    round_rows = []

    def aligned_iter():
        for idx, path in enumerate(cluster_paths, start=1):
            subtomogram_array = load_mrc(path, permissive=False)
            subtomogram_volume = Volume(subtomogram_array)

            tic = time.perf_counter()
            _, recovered_rotation = align_BO(reference_volume, subtomogram_volume, para)
            toc = time.perf_counter()

            row = {
                "round": round_idx,
                "subtomogram_id": int(path.stem),
                "file_name": path.name,
                "runtime_s": float(toc - tic),
            }
            row.update(flatten_rotation(recovered_rotation))
            round_rows.append(row)

            if verbose_every and (idx % verbose_every == 0 or idx == len(cluster_paths)):
                print(f"  Round {round_idx}: aligned {idx}/{len(cluster_paths)} subtomograms")

            yield rotate_array_with_matrix(subtomogram_array, recovered_rotation)

    next_reference = compute_average(aligned_iter())
    return next_reference, round_rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run iterative BOTalign alignment on Cluster_0 and save outputs for notebook visualization."
    )
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-rounds", type=int, default=1)
    parser.add_argument("--stop-after-round", type=int, default=None)
    parser.add_argument("--n-subtomograms", type=int, default=200)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--max-files-for-debug", type=int, default=None)
    parser.add_argument("--loss-type", choices=["wemd", "eu"], default="wemd")
    parser.add_argument("--downsample", type=int, default=91)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--verbose-every", type=int, default=10)
    parser.add_argument("--repair-headers", dest="repair_headers", action="store_true")
    parser.add_argument("--no-repair-headers", dest="repair_headers", action="store_false")
    parser.set_defaults(repair_headers=True)
    return parser


def main():
    args = build_arg_parser().parse_args()

    cluster_dir = args.cluster_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    para = [args.loss_type, args.downsample, args.iterations, args.refine]

    cluster_paths = get_cluster_paths(cluster_dir, args.n_subtomograms)
    subset_size = args.subset_size if args.subset_size is not None else args.max_files_for_debug
    if subset_size is not None:
        cluster_paths = cluster_paths[:subset_size]

    print(f"Cluster dir: {cluster_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Using {len(cluster_paths)} subtomograms")
    print(f"BOTalign parameters: {para}")

    if args.repair_headers:
        print("Repairing MRC headers in place...")
        repaired_files, repair_failures = repair_mrc_headers(cluster_paths)
        print(f"Headers repaired: {len(repaired_files)}")
        if repair_failures:
            print("Header repair failures:")
            print(pd.DataFrame(repair_failures).to_string(index=False))
            raise RuntimeError("Header repair failed for one or more files.")

    initial_reference, reference_stats_rows = ensure_initial_reference(cluster_paths, output_dir)
    current_reference, start_round, metrics_rows, final_rotations_rows, existing_reference_stats_rows = prepare_run_state(
        initial_reference, output_dir
    )
    if existing_reference_stats_rows:
        reference_stats_rows = existing_reference_stats_rows

    end_round = args.n_rounds if args.stop_after_round is None else min(args.stop_after_round, args.n_rounds)
    if start_round > end_round:
        print(f"Nothing to do. Latest completed round is {start_round - 1}, target stop round is {end_round}.")
        return

    for round_idx in range(start_round, end_round + 1):
        print(f"Starting round {round_idx}/{args.n_rounds}")
        current_reference, round_rows = align_round(
            current_reference,
            cluster_paths,
            para,
            round_idx,
            verbose_every=args.verbose_every,
        )

        reference_path = save_round_reference(output_dir, round_idx, current_reference)
        reference_stats_rows.append(reference_stats(current_reference, round_idx))

        metrics_rows.extend(round_rows)
        final_rotations_rows = list(round_rows)

        metrics_df = pd.DataFrame(metrics_rows).sort_values(["round", "subtomogram_id"]).reset_index(drop=True)
        final_rotations_df = pd.DataFrame(final_rotations_rows).sort_values("subtomogram_id").reset_index(drop=True)
        reference_stats_df = pd.DataFrame(reference_stats_rows)
        round_runtime_summary = (
            metrics_df.groupby("round", as_index=False)["runtime_s"]
            .mean()
            .rename(columns={"runtime_s": "mean_runtime_s"})
        )

        save_rotation_table(metrics_df, output_dir / "alignment_metrics.csv")
        save_rotation_table(final_rotations_df, output_dir / "final_rotations.csv")
        reference_stats_df.to_csv(output_dir / "reference_stats.csv", index=False)
        round_runtime_summary.to_csv(output_dir / "round_runtime_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "round": round_idx,
                    "status": "round_complete",
                    "processed_subtomograms": len(cluster_paths),
                    "total_subtomograms": len(cluster_paths),
                    "reference_path": str(reference_path),
                    "rotations_path": str(output_dir / "final_rotations.csv"),
                    "metrics_path": str(output_dir / "alignment_metrics.csv"),
                }
            ]
        ).to_csv(output_dir / "current_progress.csv", index=False)

        print(f"Saved round {round_idx} reference to: {reference_path}")
        print(f"Round {round_idx} mean runtime (s): {final_rotations_df['runtime_s'].mean():.4f}")

    final_reference_path = save_mrc(output_dir / "final_reference.mrc", current_reference)
    print(f"Saved final reference to: {final_reference_path}")


if __name__ == "__main__":
    main()
