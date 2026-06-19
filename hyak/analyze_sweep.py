"""Pull the Q-OIL transport sweep from W&B and plot results.

Dedups preemption-restarted runs to each config's furthest attempt (max
learner_step). A config is "done" if its best attempt passed >40k learner steps
(per the run budget); non-done configs are still-improving lower bounds.

  python hyak/analyze_sweep.py
"""
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT = "pranavnt", "wrl-qoil-sweep"
GROUP = sys.argv[1] if len(sys.argv) > 1 else "transport-sweep-v1"   # e.g. transport-sweep-v2
DONE_LS = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000          # "done" learner-step bar
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sweep_plots", GROUP)
os.makedirs(OUT, exist_ok=True)
print(f"analyzing group {GROUP} -> {OUT}")

api = wandb.Api()
runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}))
print(f"fetched {len(runs)} runs")

# ---- dedup: per config keep the attempt that reached the most learner steps --
by_cfg = defaultdict(list)
for r in runs:
    c = r.config
    key = (c.get("edit_scale"), c.get("bc_weight"), c.get("bonus"), c.get("seed"))
    if None in key:
        continue
    by_cfg[key].append(r)

rows = []
for key, rs in by_cfg.items():
    es, bc, bonus, seed = key
    best = max(rs, key=lambda r: r.summary.get("learner_step", 0) or 0)
    ls = best.summary.get("learner_step", 0) or 0
    # UNKEYED scan_history: keyed scan_history(keys=[...]) silently drops the
    # sparse eval rows for some runs; history() subsamples them out. Iterating all
    # rows and filtering is the only reliable way to recover every eval point.
    traj = [(row.get("learner_step") or 0, row["eval/success"])
            for row in best.scan_history()
            if row.get("eval/success") is not None]
    traj.sort()
    best_eval = max((s for _, s in traj), default=np.nan)
    final_eval = traj[-1][1] if traj else np.nan
    rows.append(dict(edit_scale=es, bc_weight=bc, bonus=bonus, seed=seed,
                     best_ls=ls, done=ls > DONE_LS, best_eval=best_eval,
                     final_eval=final_eval, n_evals=len(traj), traj=traj,
                     name=best.name))

n_done = sum(r["done"] for r in rows)
print(f"{len(rows)} configs, {n_done} done (>{DONE_LS//1000}k learner steps)")

# ---- CSV ------------------------------------------------------------------
csv = os.path.join(OUT, "sweep_results.csv")
with open(csv, "w") as f:
    f.write("edit_scale,bc_weight,bonus,seed,best_ls,done,best_eval,final_eval,n_evals,name\n")
    for r in sorted(rows, key=lambda x: -(x["best_eval"] if np.isfinite(x["best_eval"]) else -1)):
        f.write(f"{r['edit_scale']},{r['bc_weight']},{r['bonus']},{r['seed']},"
                f"{r['best_ls']},{int(r['done'])},{r['best_eval']:.3f},"
                f"{r['final_eval']:.3f},{r['n_evals']},{r['name']}\n")
print("wrote", csv)

# ================= Figure 1: coverage / progress ===========================
fig, ax = plt.subplots(figsize=(7, 11))
srt = sorted(rows, key=lambda r: r["best_ls"])
y = range(len(srt))
colors = ["#2a9d8f" if r["done"] else "#e9c46a" for r in srt]
ax.barh(list(y), [r["best_ls"] for r in srt], color=colors)
ax.axvline(DONE_LS, color="crimson", ls="--", lw=1, label=f"done = {DONE_LS//1000}k")
ax.set_yticks(list(y))
ax.set_yticklabels([f"es{r['edit_scale']} bc{r['bc_weight']} b{r['bonus']} s{r['seed']}"
                    for r in srt], fontsize=5)
ax.set_xlabel("best attempt: learner steps reached")
ax.set_title(f"Sweep coverage — {n_done}/{len(rows)} configs done (>{DONE_LS//1000}k)")
ax.legend(loc="lower right")
plt.tight_layout()
f1 = os.path.join(OUT, "1_coverage.png")
plt.savefig(f1, dpi=130); plt.close()

# ================= Figure 2: best eval vs each hyperparam ===================
# done configs as solid (real results); in-progress as faint (lower bounds).
# axis values are DERIVED from the data so any sweep grid plots correctly.
axes_specs = [(f, sorted({r[f] for r in rows}))
              for f in ("edit_scale", "bc_weight", "bonus")]
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, (field, vals) in zip(axs, axes_specs):
    for i, v in enumerate(vals):
        grp = [r for r in rows if r[field] == v and np.isfinite(r["best_eval"])]
        for r in grp:
            jitter = (np.random.RandomState(hash((field, r["name"])) % 2**31).rand() - 0.5) * 0.18
            ax.scatter(i + jitter, r["best_eval"],
                       c="#2a9d8f" if r["done"] else "#e9c46a",
                       s=55 if r["done"] else 28,
                       edgecolors="k" if r["done"] else "none", linewidths=0.4,
                       alpha=0.95 if r["done"] else 0.5, zorder=3 if r["done"] else 2)
        done_vals = [r["best_eval"] for r in grp if r["done"]]
        if done_vals:
            ax.scatter(i, np.mean(done_vals), marker="_", s=2200, c="crimson",
                       lw=2.5, zorder=4)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals)
    ax.set_xlabel(field); ax.set_ylim(-0.03, 1.0); ax.grid(axis="y", alpha=0.3)
axs[0].set_ylabel("best eval success")
from matplotlib.lines import Line2D
axs[2].legend(handles=[
    Line2D([0],[0], marker="o", ls="", mfc="#2a9d8f", mec="k", label=f"done (>{DONE_LS//1000}k)"),
    Line2D([0],[0], marker="o", ls="", mfc="#e9c46a", mec="none", label="in-progress (lower bound)"),
    Line2D([0],[0], marker="_", ls="", mec="crimson", label="done mean")], fontsize=8)
fig.suptitle("Best eval success vs hyperparameter (each point = one config/seed)")
plt.tight_layout()
f2 = os.path.join(OUT, "2_by_hyperparam.png")
plt.savefig(f2, dpi=130); plt.close()

# ================= Figure 3: eval trajectories (top configs) ===============
# longest attempt per config; show the top-N by best eval (regardless of done).
TOPN = 12
fig, ax = plt.subplots(figsize=(11, 6.5))
top = sorted([r for r in rows if r["traj"]],
             key=lambda r: -r["best_eval"])[:TOPN]
cmap = plt.cm.viridis(np.linspace(0, 0.92, max(len(top), 1)))
for r, col in zip(top, cmap):
    xs = [a for a, _ in r["traj"]]; ys = [b for _, b in r["traj"]]
    mark = "-o" if r["done"] else "--o"
    ax.plot(xs, ys, mark, ms=3, color=col, lw=1.4,
            label=f"es{r['edit_scale']} bc{r['bc_weight']} b{r['bonus']} s{r['seed']} "
                  f"({r['best_eval']:.0%}{'' if r['done'] else ', '+str(r['best_ls']//1000)+'k'})")
ax.axhline(0.34, color="gray", ls=":", lw=1, label="base DP ~34%")
ax.set_xlabel("learner step"); ax.set_ylabel("eval success")
ax.set_title(f"Eval trajectories — top {len(top)} configs by best eval "
             f"(solid=done >{DONE_LS//1000}k, dashed=in-progress)")
ax.set_ylim(-0.03, 1.0); ax.grid(alpha=0.3)
ax.legend(fontsize=7, ncol=2, loc="lower right")
plt.tight_layout()
f3 = os.path.join(OUT, "3_trajectories.png")
plt.savefig(f3, dpi=130); plt.close()

print("wrote", f1, f2, f3, sep="\n")

# ---- console: top configs -------------------------------------------------
print("\n=== top configs by best eval (done first) ===")
for r in sorted(rows, key=lambda x: (x["done"], x["best_eval"] if np.isfinite(x["best_eval"]) else -1),
                reverse=True)[:12]:
    tag = "DONE" if r["done"] else f"{r['best_ls']//1000}k"
    print(f"  es{r['edit_scale']} bc{r['bc_weight']} b{r['bonus']} s{r['seed']}  "
          f"best={r['best_eval']:.0%} final={r['final_eval']:.0%}  [{tag}]")
