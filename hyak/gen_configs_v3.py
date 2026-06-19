"""Q-OIL transport sweep v3 -> hyak/configs_v3.txt.

Isolates the OPTIMISM effect: fix the winning edit_scale=0.15, bc_weight=0.1
(from v1/v2) and sweep intervention_bonus only, now including a strong 2.0, with
5 seeds for tight error bars. Removes the bc=0 / edit_scale confounds that
flattened the bonus marginal in v2. Each line: `edit_scale bc_weight bonus seed`.

    python hyak/gen_configs_v3.py
    # launch (separate wandb group; 20 configs, 34k steps each):
    CONFIGS=$WRL/hyak/configs_v3.txt WANDB_GROUP=transport-sweep-v3 MAX_STEPS=34000 \
        sbatch --time=5:00:00 --array=1-20 hyak/sweep.sbatch
"""
import itertools
import os

EDIT_SCALES = [0.15]
BC_WEIGHTS = [0.1]
BONUSES = [0.0, 0.1, 0.4, 2.0]
SEEDS = [0, 1, 2, 3, 4]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs_v3.txt")
rows = list(itertools.product(EDIT_SCALES, BC_WEIGHTS, BONUSES, SEEDS))
with open(out, "w") as f:
    for es, bc, b, s in rows:
        f.write(f"{es} {bc} {b} {s}\n")
print(f"wrote {len(rows)} configs -> {out}")
print(f"launch:  CONFIGS=$WRL/hyak/configs_v3.txt WANDB_GROUP=transport-sweep-v3 "
      f"MAX_STEPS=34000 sbatch --time=5:00:00 --array=1-{len(rows)} hyak/sweep.sbatch")
