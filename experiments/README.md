# experiments/

Each subdirectory here is one experiment run.

## Naming convention

```
exp{NNN}_{method}_{short_description}/
```

Examples:
- `exp001_empot_n40_first_do_log_bug/`
- `exp004_empot_lowpass_prefilter/`
- `exp005_botalign_eman2_seed/`

## What goes inside each folder

| File | Description |
|------|-------------|
| `config.yaml` | **Every parameter** used in this run — exact reproducibility |
| `notes.md` | What you found, why it worked or failed, next step |
| `run.log` | Console output captured from the run |
| `round_metrics_summary.csv` | Per-round metrics (L2 change, UGW dist, runtime) |
| `metrics_details_round_XX.csv` | Per-subtomogram metrics per round |
| `slices_round_XX.png` | Orthoslice visualizations |

## What does NOT go here

- `.mrc` volume outputs (too large — gitignored)
- `.hdf` files
- Virtual environments

## Template for a new experiment

```bash
# 1. Create folder
mkdir -p experiments/exp004_my_experiment

# 2. Write config before running
cat > experiments/exp004_my_experiment/config.yaml << 'EOF'
experiment_id: exp004
date: YYYY-MM-DD
method: empot
description: "One sentence description of what this tests"

data:
  cluster_dir: /abs/path/to/Data/0.1/Cluster_0
  n_sbtms: 40
  selection_mode: first   # or random

params:
  n_rounds: 5
  threshold: 0.7
  num_points: 500
  eps: 10000
  seed: 42

notes: ""
EOF

# 3. Run — point output to this folder
python EMPOT/src/run_iterative_alignment.py \
    --cluster_dir /abs/path/... \
    --output_base_dir experiments/exp004_my_experiment \
    --n_sbtms 40 --n_rounds 5

# 4. Write notes.md immediately after
```
