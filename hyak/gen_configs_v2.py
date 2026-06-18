"""Q-OIL transport sweep v2 -> hyak/configs_v2.txt.

Follow-up to v1: probe optimism (intervention_bonus) wider — including 0 (no
optimism) and 0.4 — and bc_weight=0 (no BC reg). edit_scale narrowed to v1's
two viable values (0.35 was clearly worst). Each line: `edit_scale bc_weight
intervention_bonus seed`.

    python hyak/gen_configs_v2.py
    # launch (separate wandb group; v2 has 60 configs):
    CONFIGS=$WRL/hyak/configs_v2.txt WANDB_GROUP=transport-sweep-v2 \
        sbatch --array=1-60 hyak/sweep.sbatch
"""
import itertools
import os

EDIT_SCALES = [0.15, 0.25]
BC_WEIGHTS = [0.0, 0.1]
BONUSES = [0.0, 0.05, 0.1, 0.2, 0.4]
SEEDS = [0, 1, 2]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs_v2.txt")
rows = list(itertools.product(EDIT_SCALES, BC_WEIGHTS, BONUSES, SEEDS))
with open(out, "w") as f:
    for es, bc, b, s in rows:
        f.write(f"{es} {bc} {b} {s}\n")
print(f"wrote {len(rows)} configs -> {out}")
print(f"launch:  CONFIGS=$WRL/hyak/configs_v2.txt WANDB_GROUP=transport-sweep-v2 "
      f"sbatch --array=1-{len(rows)} hyak/sweep.sbatch")
