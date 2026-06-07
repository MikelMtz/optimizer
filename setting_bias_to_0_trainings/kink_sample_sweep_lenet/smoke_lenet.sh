#!/bin/bash
#SBATCH --job-name=lenet_smoke
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=logs/lenet_smoke_%j.out
#SBATCH --error=logs/lenet_smoke_%j.err
#
# Smoke-test for the kink+lenet sweep.  Runs 4 representative cases
# (guess+width, SGD+width, guess+depth, SGD+depth) with a tiny
# model_count and target=1, then extrapolates to production runtimes.
#
# Usage (SLURM):  sbatch smoke_lenet.sh
# Usage (local):  bash  smoke_lenet.sh

export CUDA_VISIBLE_DEVICES=""

SMOKE_MCTBS=5000   # intentionally small; production uses up to 320 000
SMOKE_DIR="output/smoke_lenet"
mkdir -p "$SMOKE_DIR" logs

# ── helper ────────────────────────────────────────────────────────────────────
# Run one smoke case, print full python output, and save wall-clock seconds
# to $SMOKE_DIR/<label>.elapsed
smoke() {
    local label="$1" config="$2"; shift 2
    echo ""
    echo "━━━ SMOKE [$label] ━━━"
    local t0=$SECONDS
    python train_distributed_bias_0.py \
        --config "$config" \
        "$@" \
        --model.model_count_times_batch_size $SMOKE_MCTBS \
        --distributed.num_samples "16" \
        --distributed.target_model_count_subrun 1 \
        --output.target_model_count 1 \
        --output.folder "$SMOKE_DIR/$label"
    local elapsed=$(( SECONDS - t0 ))
    echo "$elapsed" > "$SMOKE_DIR/${label}.elapsed"
    echo "━━━ SMOKE [$label] done in ${elapsed}s ━━━"
}

# ── 4 representative cases ────────────────────────────────────────────────────
#
#  guess + width  (width=1, depth=4 default)
#    → calibrates all 7-width guess tasks; worst task is w=0.25 (MCTBS 320k)
smoke guess_width_w1 \
    configs/kink_sweep/kink_lenet_width_sweep_guess.yaml \
    --model.lenet.width 1 \
    --model.init uniform

#  SGD + width  (width=1, depth=4 default)
#    → calibrates all 7-width SGD tasks; worst task is w=0.25 (MCTBS 240k)
smoke sgd_width_w1 \
    configs/kink_sweep/kink_lenet_width_sweep_sgd.yaml \
    --model.lenet.width 1 \
    --model.init uniform

#  guess + depth  (depth=4, worst-case network size → upper bound for all depths)
smoke guess_depth_d4 \
    configs/kink_sweep/kink_lenet_depth_sweep_guess.yaml \
    --model.lenet.layers 4 \
    --model.init uniform

#  SGD + depth  (depth=4, worst-case → upper bound for all depths)
smoke sgd_depth_d4 \
    configs/kink_sweep/kink_lenet_depth_sweep_sgd.yaml \
    --model.lenet.layers 4 \
    --model.init uniform

# ── read timings ──────────────────────────────────────────────────────────────
T_GW=$(cat "$SMOKE_DIR/guess_width_w1.elapsed"  2>/dev/null || echo 0)
T_SW=$(cat "$SMOKE_DIR/sgd_width_w1.elapsed"    2>/dev/null || echo 0)
T_GD=$(cat "$SMOKE_DIR/guess_depth_d4.elapsed"  2>/dev/null || echo 0)
T_SD=$(cat "$SMOKE_DIR/sgd_depth_d4.elapsed"    2>/dev/null || echo 0)

# ── estimate production runtimes ─────────────────────────────────────────────
python3 - "$SMOKE_MCTBS" "$T_GW" "$T_SW" "$T_GD" "$T_SD" <<'PYEOF'
import sys, math

SMOKE_MCTBS  = int(sys.argv[1])
t_gw = int(sys.argv[2])   # guess + width=1  (one cell, target=1)
t_sw = int(sys.argv[3])   # SGD   + width=1  (one cell, target=1)
t_gd = int(sys.argv[4])   # guess + depth=4  (one cell, target=1)
t_sd = int(sys.argv[5])   # SGD   + depth=4  (one cell, target=1)

TARGET        = 100     # perfect models per sample size in production
N_SAMPLE_SIZES = 9      # 2,4,6,8,12,16,20,26,30
CELL_TIMEOUT  = 2 * 3600  # 2h per cell (hard cap in training code)
BATCH_SIZE_SGD = 2
FALLBACK_MAX   = 30000
SGD_EPOCHS_OLD = 30     # smoke was run with this
SGD_EPOCHS_NEW = 15     # reduced after smoke result

# Production MCTBS tables (from kink_sample_sweep_lenet.sh)
WIDTHS     = [0.25, 0.5, 1, 2, 4, 8, 16]
MCTBS_GW   = {0.25:320000, 0.5:160000, 1:160000, 2:80000, 4:40000, 8:16000, 16:8000}
MCTBS_SW   = {0.25:240000, 0.5:120000, 1:120000, 2:16000, 4:8000,  8:4000,  16:2000}
DEPTHS     = [1, 2, 3, 4]
MCTBS_GD   = 80000
MCTBS_SD   = 60000


def fmt(s):
    s = max(int(s), 1)
    if s >= 3600: return f"{s//3600}h {(s%3600)//60:02d}m"
    return f"{s//60}m {s%60:02d}s"

def slurm_time(s, safety=1.1):
    s = int(s * safety) + 3600  # +1h startup/overhead
    return f"{s//3600:02d}:{(s%3600)//60:02d}:00"

# ── GUESS estimator ───────────────────────────────────────────────────────────
# Key insight from smoke: smoke already ran until 1 perfect model was found.
# Larger MCTBS means larger batches → proportionally more hits per batch
# → wall time to find N perfect ≈ t_smoke_per_cell × N, independent of MCTBS.
# Per-task estimate = t_smoke × target, capped by CELL_TIMEOUT per sample size.
def est_guess(t_smoke_cell):
    t_per_sample = min(t_smoke_cell * TARGET, CELL_TIMEOUT)
    return t_per_sample * N_SAMPLE_SIZES

# ── SGD estimator ─────────────────────────────────────────────────────────────
# Key insight from smoke: virtually ALL trained models reach acc=1.0.
# → 1 training batch of model_count_prod >> target perfect models.
# Wall time scales with model_count (parallelism) and epoch count.
# model_count is capped at FALLBACK_MAX=30000.
def est_sgd(t_smoke_cell, mctbs_prod):
    model_count_smoke = SMOKE_MCTBS // BATCH_SIZE_SGD    # 2500
    model_count_prod  = min(mctbs_prod // BATCH_SIZE_SGD, FALLBACK_MAX)
    epoch_factor      = SGD_EPOCHS_NEW / SGD_EPOCHS_OLD
    t_per_batch_prod  = t_smoke_cell * (model_count_prod / model_count_smoke) * epoch_factor
    # batches needed per sample size: 1 batch gives model_count_prod >> target
    batches_needed    = max(1, math.ceil(TARGET / model_count_prod))
    t_per_sample      = min(t_per_batch_prod * batches_needed, CELL_TIMEOUT)
    return t_per_sample * N_SAMPLE_SIZES


SEP = "=" * 68
print(f"\n{SEP}")
print("  LENET KINK SWEEP — CORRECTED PRODUCTION RUNTIME ESTIMATES")
print(f"  smoke MCTBS={SMOKE_MCTBS}  |  SGD epochs {SGD_EPOCHS_OLD}→{SGD_EPOCHS_NEW}  |  target={TARGET}/sample_size")
print(SEP)

# ── guess width sweep ─────────────────────────────────────────────────────────
print(f"\n  Width sweep — guess optimizer  (smoke/cell: {fmt(t_gw)})")
print(f"  {'width':>6}   {'t/sample_size':>14}   {'est/task':>10}   note")
tw_g = []
for w in WIDTHS:
    t = est_guess(t_gw)  # same for all widths (MCTBS cancels in wall time)
    tw_g.append(t)
    cap = "⚑ CELL_TIMEOUT cap" if t_gw * TARGET >= CELL_TIMEOUT else ""
    print(f"  {w:>6.2f}   {fmt(t//N_SAMPLE_SIZES):>14}   {fmt(t):>10}   {cap}")
wall_gw  = max(tw_g)
total_gw = sum(tw_g) * 4  # ×4 inits
print(f"  → worst task: {fmt(wall_gw)}  |  total compute (4 inits): {fmt(total_gw)}")

# ── SGD width sweep ───────────────────────────────────────────────────────────
print(f"\n  Width sweep — SGD optimizer    (smoke/batch: {fmt(t_sw)}, now {SGD_EPOCHS_NEW} epochs)")
print(f"  {'width':>6}   {'model_count':>11}   {'t/sample_size':>14}   {'est/task':>10}")
tw_s = []
for w in WIDTHS:
    mc = min(MCTBS_SW[w] // BATCH_SIZE_SGD, FALLBACK_MAX)
    t  = est_sgd(t_sw, MCTBS_SW[w])
    tw_s.append(t)
    print(f"  {w:>6.2f}   {mc:>11,}   {fmt(t//N_SAMPLE_SIZES):>14}   {fmt(t):>10}")
wall_sw  = max(tw_s)
total_sw = sum(tw_s) * 4
print(f"  → worst task: {fmt(wall_sw)}  |  total compute (4 inits): {fmt(total_sw)}")

# ── guess depth sweep ─────────────────────────────────────────────────────────
print(f"\n  Depth sweep — guess optimizer  (smoke/cell: {fmt(t_gd)}, MCTBS {MCTBS_GD:,})")
print(f"  {'depth':>6}   {'est/task':>10}   note")
td_g = []
for d in DEPTHS:
    t = est_guess(t_gd)
    td_g.append(t)
    cap = "⚑ CELL_TIMEOUT cap" if t_gd * TARGET >= CELL_TIMEOUT else ""
    print(f"  {d:>6}   {fmt(t):>10}   {cap}")
wall_gd  = max(td_g)
total_gd = sum(td_g) * 4
print(f"  → worst task: {fmt(wall_gd)}  |  total compute (4 inits): {fmt(total_gd)}")

# ── SGD depth sweep ───────────────────────────────────────────────────────────
print(f"\n  Depth sweep — SGD optimizer    (smoke/batch: {fmt(t_sd)}, now {SGD_EPOCHS_NEW} epochs, MCTBS {MCTBS_SD:,})")
print(f"  {'depth':>6}   {'model_count':>11}   {'est/task':>10}   note")
td_s = []
for d in DEPTHS:
    mc = min(MCTBS_SD // BATCH_SIZE_SGD, FALLBACK_MAX)
    t  = est_sgd(t_sd, MCTBS_SD)
    td_s.append(t)
    note = "" if d == 4 else "(d<4 is faster)"
    print(f"  {d:>6}   {mc:>11,}   {fmt(t):>10}   {note}")
wall_sd  = max(td_s)
total_sd = sum(td_s) * 4
print(f"  → worst task: {fmt(wall_sd)}  |  total compute (4 inits): {fmt(total_sd)}")

# ── summary ───────────────────────────────────────────────────────────────────
total_all  = total_gw + total_sw + total_gd + total_sd
worst_task = max(wall_gw, wall_sw, wall_gd, wall_sd)

print(f"\n{SEP}")
print(f"  TOTAL compute (all 88 tasks):   {fmt(total_all)}")
print(f"  WORST single task:              {fmt(worst_task)}")
print(f"  Hard ceiling (9 cells × 2h):    18h 00m  [CELL_TIMEOUT in code]")
print(f"  Recommended #SBATCH --time:     {slurm_time(min(worst_task, 18*3600))}  (ceiling + 10% + 1h overhead)")
print(f"\n  Methodology corrections vs previous run:")
print(f"    · Guess: wall time ∝ target only (larger MCTBS → more hits/batch,")
print(f"      cancels the time increase → MCTBS does NOT compound runtime).")
print(f"    · SGD:   ~100% of models reach acc=1.0 → 1 batch ≥ target;")
print(f"      scales with model_count (capped at {FALLBACK_MAX:,}) × epoch reduction.")
print(f"    · Per-task ceiling = {N_SAMPLE_SIZES} sample_sizes × CELL_TIMEOUT=2h = 18h.")
print(f"{SEP}\n")
PYEOF
