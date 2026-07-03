# Lab Notebook — Subtomogram Alignment Project

> **How to use this file:**  
> Append a new entry **every day** you work on the project.  
> Keep it honest — failed experiments are just as important as successes.  
> Copy key numbers to the Google Sheet after each run.

---

## Entry Template

```
## YYYY-MM-DD

### What I worked on
-

### Experiments run
| exp_id | method | n_sbtms | key params | outcome |
|--------|--------|---------|------------|---------|
| | | | | |

### Findings
-

### Decisions made
-

### Open questions / Next steps
-
```

---

## 2026-07-03

### What I worked on
- Reviewed updated `PROJECT_CONTEXT.md` — consolidated all findings from the OT failure investigation and EMAN2 baseline into the document.
- Planned repository restructuring and GitHub workflow.
- Set up `.gitignore`, `experiments/`, and `docs/` scaffold.
- Backed up entire project to `subtmgr-align-backup`.

### Experiments run
*None today — setup day.*

### Findings
- All three OT methods (AlignOT, BOTalign, EMPOT) **diverge** on SNR 0.1 data: average gets worse each round.
- Root causes (from PROJECT_CONTEXT.md investigation):
  1. **Missing wedge** (dominant) — fixed in lab frame, OT aligns wedge artifacts instead of protein.
  2. **SNR 0.1** — TRN samples noise peaks, not structure.
  3. **Bad transforms + real-space averaging** — random rotations smear the average further.
- **EMAN2 baseline converges** (scores: −0.606 → −0.672 → best at iter 3), proving the data IS alignable.
- Point-cloud OT on subsampled volumes will **never work** at SNR 0.0001.

### Decisions made
- Will NOT pursue further TRN point-cloud tuning.
- Keep separate environments for each method (alignOT, BOTalign, EMPOT) — they have conflicting deps.
- Use `experiments/` for all future runs with `config.yaml` + `notes.md` per experiment.
- Use Google Sheet for daily result overview.
- Git branches for **code logic changes**; config files for **parameter variations**.

### Open questions / Next steps
- [ ] Get simulation metadata from project guide (tilt range, true structure, pixel size).
- [ ] Decide next experimental direction: Fourier-domain alignment? Low-pass prefilter? OT on Fourier shells?
- [ ] Set up GitHub remote and push initial commit.
- [ ] Design Google Sheet columns.

---

## Experiment Log Index

| exp_id | date | branch | method | snr | n_sbtms | key params | converged? | conclusion |
|--------|------|--------|--------|-----|---------|------------|-----------|------------|
| exp001 | 2026-06-23 | — | EMPOT | 0.1 | 40 | eps=2000, 5 rounds | ❌ No (do_log bug → all zeros) | Bug: do_log=False in TRN returned zero point clouds |
| exp002 | 2026-06-23 | — | EMPOT | 0.1 | 40 | eps=10000, 5 rounds | ❌ No (diverges) | Average degrades each round; TRN sampling noise not structure |
| exp003 | 2026-06-25 | — | EMPOT | 0.1 | 5 | eps=10000, first valid run | ❌ No | Confirmed divergence — root cause: missing wedge + SNR |

---
