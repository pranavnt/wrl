"""Q-OIL transport sweep v5 -> hyak/configs_v5.txt.

Zoom into the LOW-optimism regime. v4 (bonus 0..2.0) showed the optimism effect
is an inverted-U that peaks at small bonus and decays badly for large bonus
(bc=0.25: 57% @ 0.4 -> 28% @ 2.0), while bc=0 collapses. v5 samples below v4's
0.4 grid point to locate the actual peak, on the two live BC rows only.

    edit_scale = 0.15                       (fixed; v1-v4 winner)
    bc_weight  in {0.1, 0.25}               (bc=0 collapses, dropped)
    bonus      in {0.025, 0.05, 0.1, 0.2}   (sub-0.4, finer)
    seed       in {0, 1, 2}
    -> 2 x 4 x 3 = 24 configs

Each line: `edit_scale bc_weight bonus seed`.

    python hyak/gen_configs_v5.py
    # launch (separate wandb group; 24 configs, 34k steps each, 9h wall):
    CONFIGS=$WRL/hyak/configs_v5.txt WANDB_GROUP=transport-sweep-v5 MAX_STEPS=34000 \
        sbatch --time=9:00:00 --array=1-24 hyak/sweep.sbatch
"""
import itertools
import os

EDIT_SCALES = [0.15]
BC_WEIGHTS = [0.1, 0.25]
BONUSES = [0.025, 0.05, 0.1, 0.2]
SEEDS = [0, 1, 2]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs_v5.txt")
rows = list(itertools.product(EDIT_SCALES, BC_WEIGHTS, BONUSES, SEEDS))
with open(out, "w") as f:
    for es, bc, b, s in rows:
        f.write(f"{es} {bc} {b} {s}\n")
print(f"wrote {len(rows)} configs -> {out}")
print(f"launch:  CONFIGS=$WRL/hyak/configs_v5.txt WANDB_GROUP=transport-sweep-v5 "
      f"MAX_STEPS=34000 sbatch --time=9:00:00 --array=1-{len(rows)} hyak/sweep.sbatch")
