"""Generate the Q-OIL sweep grid -> hyak/configs.txt (one job per line).

Each line: `edit_scale bc_weight intervention_bonus seed`. The sbatch array
maps SLURM_ARRAY_TASK_ID -> line N. Edit the arrays below to resize the sweep;
re-run `python hyak/gen_configs.py` and update --array=1-<N> in sweep.sbatch.
"""

import itertools
import os

EDIT_SCALES = [0.15, 0.25, 0.35]
BC_WEIGHTS = [0.1, 0.25, 0.5]
BONUSES = [0.05, 0.1, 0.2]
SEEDS = [0, 1, 2]

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "configs.txt")
rows = list(itertools.product(EDIT_SCALES, BC_WEIGHTS, BONUSES, SEEDS))
with open(out, "w") as f:
    for es, bc, b, s in rows:
        f.write(f"{es} {bc} {b} {s}\n")
print(f"wrote {len(rows)} configs -> {out}")
print(f"set sweep.sbatch:  #SBATCH --array=1-{len(rows)}%16")
