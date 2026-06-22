"""Q-OIL transport sweep v4 -> hyak/configs_v4.txt.

Proper bonus x bc_weight grid. v3 showed a monotonic optimism trend
(bonus 0->2 lifted best eval ~39%->54%) but 18/20 runs died at the 5h wall
(~22k steps) so the magnitude is untrustworthy. v4 fixes that: 9h wall fits the
full 34k budget on the slow ckpt nodes (~4.5k steps/h), and fills in the bonus
axis between 0 and 2. edit_scale stays at the v1/v2/v3 winner (0.15).

    edit_scale = 0.15            (fixed)
    bc_weight  in {0, 0.1, 0.25}
    bonus      in {0, 0.4, 0.8, 1.2, 1.6, 2.0}
    seed       in {0, 1, 2}
    -> 3 x 6 x 3 = 54 configs

Each line: `edit_scale bc_weight bonus seed`.

    python hyak/gen_configs_v4.py
    # launch (separate wandb group; 54 configs, 34k steps each, 9h wall):
    CONFIGS=$WRL/hyak/configs_v4.txt WANDB_GROUP=transport-sweep-v4 MAX_STEPS=34000 \
        sbatch --time=9:00:00 --array=1-54 hyak/sweep.sbatch
"""
import itertools
import os

EDIT_SCALES = [0.15]
BC_WEIGHTS = [0.0, 0.1, 0.25]
BONUSES = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
SEEDS = [0, 1, 2]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs_v4.txt")
rows = list(itertools.product(EDIT_SCALES, BC_WEIGHTS, BONUSES, SEEDS))
with open(out, "w") as f:
    for es, bc, b, s in rows:
        f.write(f"{es} {bc} {b} {s}\n")
print(f"wrote {len(rows)} configs -> {out}")
print(f"launch:  CONFIGS=$WRL/hyak/configs_v4.txt WANDB_GROUP=transport-sweep-v4 "
      f"MAX_STEPS=34000 sbatch --time=9:00:00 --array=1-{len(rows)} hyak/sweep.sbatch")
