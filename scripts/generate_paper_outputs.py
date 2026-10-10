#!/usr/bin/env python3
"""
Generate the paper figures and the family-comparison table from the CSV files
produced by src/collaborative_forgetting_experiments.py.

Expected inputs in --input-dir (default: results/):
  results_depth_k.csv
  results_lambda.csv
  results_scalability.csv
  results_families.csv
  results_ilp_scalability.csv  (optional, adds the exact solver to Fig. 4)

Outputs in --output-dir (default: figures/):
  fig_trap_depth_cost.pdf/.png
  fig_lambda_sensitivity.pdf/.png
  fig_scalability_affected_region.pdf/.png
  table_families.tex
  summary_results.txt
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

matplotlib.rcParams["font.size"] = 9
COLORS = {"greedy": "0.45", "downstream_k1": "#1f77b4",
          "downstream_k2": "#ff7f0e", "downstream_k3": "#2ca02c"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mean(rows, column):
    vals = [float(r[column]) for r in rows]
    return statistics.mean(vals) if vals else float("nan")


def selected(rows, **conds):
    out = []
    for r in rows:
        ok = True
        for k, v in conds.items():
            rv = r[k]
            if rv == "" and not isinstance(v, str):
                ok = False  # e.g. empty lambda for cascade/greedy rows
                break
            if isinstance(v, int):
                ok &= int(float(rv)) == v
            elif isinstance(v, float):
                ok &= abs(float(rv) - v) < 1e-12
            else:
                ok &= rv == v
        if ok:
            out.append(r)
    return out


def make_trap_depth_figure(rows, outdir):
    algs = ["greedy", "downstream_k1", "downstream_k2", "downstream_k3"]
    labels = ["Greedy", "1-step", "2-step", "3-step"]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    depths = [1, 2, 3]

    plt.figure(figsize=(3.4, 2.0))
    for alg, label, ls, mk in zip(algs, labels, linestyles, markers):
        ys = [mean(selected(rows, algorithm=alg, trap_depth=d), "cost")
              for d in depths]
        plt.plot(depths, ys, label=label, linestyle=ls, marker=mk, color=COLORS[alg])
    plt.xlabel("Adversarial trap depth")
    plt.ylabel("Mean repair cost")
    plt.xticks(depths)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False, fontsize=7.5, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=2.0)
    plt.tight_layout()
    plt.savefig(outdir / "fig_trap_depth_cost.pdf", bbox_inches="tight")
    plt.savefig(outdir / "fig_trap_depth_cost.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_lambda_figure(rows, outdir):
    lambdas = sorted({
        float(r["lambda"]) for r in rows
        if r["algorithm"].startswith("downstream_") and r["lambda"] != ""
    })

    # Categorical x-axis: lambda values are not evenly spaced.
    xs = list(range(len(lambdas)))
    plt.figure(figsize=(3.4, 2.0))
    greedy = mean(selected(rows, algorithm="greedy"), "cost")
    plt.axhline(greedy, linestyle=(0, (4, 2)), color=COLORS["greedy"],
                linewidth=1.0, label="Greedy", zorder=1)
    for alg, label, ls, mk, ms, mfc in [
        ("downstream_k1", "1-step", "-", "s", 5, None),
        ("downstream_k2", "2-step", "-", "^", 5, "white"),
        ("downstream_k3", "3-step", ":", "D", 3.5, None),
    ]:
        ys = [mean(selected(rows, algorithm=alg, **{"lambda": lv}), "cost")
              for lv in lambdas]
        plt.plot(xs, ys, label=label, linestyle=ls, marker=mk, markersize=ms,
                 color=COLORS[alg], markerfacecolor=mfc or COLORS[alg],
                 linewidth=1.4, zorder=3 if alg == "downstream_k3" else 2)
    plt.xticks(xs, [f"{lv:g}" for lv in lambdas])
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Mean repair cost")
    plt.grid(True, axis="y", color="0.9", linewidth=0.6)
    ax = plt.gca()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.legend(frameon=False, fontsize=7.5, loc="upper right",
               bbox_to_anchor=(1.0, 0.87), ncol=2, columnspacing=1.0)
    plt.tight_layout()
    plt.savefig(outdir / "fig_lambda_sensitivity.pdf", bbox_inches="tight")
    plt.savefig(outdir / "fig_lambda_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_scalability_figure(rows, outdir, ilp_rows=None):
    agg = defaultdict(list)
    for r in rows:
        if r["algorithm"] not in ("greedy", "downstream_k1"):
            continue
        x = int(float(r["affected_size"]))
        agg[(r["algorithm"], x)].append(float(r["runtime_s"]))

    xs = sorted({x for (_, x) in agg})
    plt.figure(figsize=(3.4, 2.0))
    for alg, label, ls, mk in [
        ("greedy", "Greedy", "-", "o"),
        ("downstream_k1", "1-step", "--", "s"),
    ]:
        xvals, yvals = [], []
        for x in xs:
            vals = agg.get((alg, x))
            if vals:
                xvals.append(x)
                yvals.append(statistics.mean(vals))
        plt.plot(xvals, yvals, label=label, linestyle=ls, marker=mk, color=COLORS[alg])

    if ilp_rows:
        ilp = defaultdict(list)
        for r in ilp_rows:
            ilp[int(float(r["affected"]))].append(float(r["ilp_time"]))
        ix = sorted(ilp)
        plt.plot(ix, [statistics.mean(ilp[x]) for x in ix], label="Exact (CP-SAT)",
                 linestyle=":", marker="D", color="#2ca02c")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Affected-region size")
    plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="major", alpha=0.20)
    plt.legend(frameon=False, fontsize=7.5)
    plt.tight_layout()
    plt.savefig(outdir / "fig_scalability_affected_region.pdf", bbox_inches="tight")
    plt.savefig(outdir / "fig_scalability_affected_region.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_family_table(rows, outdir):
    algs = [
        ("cascade", r"\textsc{Cascade}"),
        ("greedy", r"\textsc{Greedy}"),
        ("downstream_k1", r"\textsc{DA}-1"),
        ("downstream_k3", r"\textsc{DA}-3"),
    ]
    families = ["LOCAL", "SHARED", "ADVERSARIAL"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Mean repair cost (and retention ratio) per graph family. "
        r"Retention is measured over $V_{\mathcal R}\setminus S$. "
        r"\textsc{Adversarial} uses trap depth $d=1$.}",
        r"\label{tab:families}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Family & \textsc{Cascade} & \textsc{Greedy} & \textsc{DA}-1 & \textsc{DA}-3 \\",
        r"\midrule",
    ]

    summary_lines = []
    for fam in families:
        cells = []
        for alg, _ in algs:
            rs = selected(rows, family=fam, algorithm=alg)
            c = mean(rs, "cost")
            ret = 100.0 * mean(rs, "retention")
            cells.append(f"{c:.2f} ({ret:.2f}\\%)")
            summary_lines.append(f"{fam:11s} {alg:15s} cost={c:.6f} retention={ret:.6f}%")
        lines.append(rf"\textsc{{{fam.title()}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    (outdir / "table_families.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="results")
    p.add_argument("--output-dir", default="figures")
    args = p.parse_args()

    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    depth = read_csv(indir / "results_depth_k.csv")
    lamb = read_csv(indir / "results_lambda.csv")
    scal = read_csv(indir / "results_scalability.csv")
    fam = read_csv(indir / "results_families.csv")

    make_trap_depth_figure(depth, outdir)
    make_lambda_figure(lamb, outdir)
    ilp_scal_path = indir / "results_ilp_scalability.csv"
    ilp_scal = read_csv(ilp_scal_path) if ilp_scal_path.exists() else None
    make_scalability_figure(scal, outdir, ilp_scal)
    summary = make_family_table(fam, outdir)

    # Depth-3 retention check used in abstract/RQ2/conclusion.
    summary.append("")
    summary.append("Depth-3 retention check (paper definition excludes S):")
    for alg in ["greedy", "downstream_k1", "downstream_k2", "downstream_k3"]:
        r = 100.0 * mean(selected(depth, algorithm=alg, trap_depth=3), "retention")
        summary.append(f"{alg:15s}: {r:.6f}%")

    # RQ1: comparison with the exact optimum (optional campaign).
    exact_path = indir / "results_exact.csv"
    if exact_path.exists():
        ex = read_csv(exact_path)
        key = lambda r: (r["seed"], r["n"], r["actors"], r["trap_depth"])
        opt = {key(r): float(r["cost"]) for r in ex if r["algorithm"] == "optimal"}
        g = [r for r in ex if r["algorithm"] == "greedy" and key(r) in opt]
        d = [r for r in ex if r["algorithm"] == "downstream_k1" and key(r) in opt]
        gaps = [float(r["gap"]) for r in g]
        summary.append("")
        summary.append(f"RQ1 exact comparison on {len(opt)} instances:")
        summary.append(f"  mean optimal cost : {statistics.mean(opt.values()):.2f}")
        summary.append(f"  mean greedy cost  : {mean(g, 'cost'):.2f}")
        summary.append(f"  mean DA-1 cost    : {mean(d, 'cost'):.2f}")
        summary.append(f"  greedy gap mean/median/max: {100*statistics.mean(gaps):.1f}% / "
                       f"{100*statistics.median(gaps):.1f}% / {100*max(gaps):.1f}%")
        summary.append(f"  greedy optimal on {sum(x < 1e-9 for x in gaps)}/{len(gaps)}; "
                       f"DA-1 optimal on {sum(abs(float(r['gap'])) < 1e-9 for r in d)}/{len(d)}")

    (outdir / "summary_results.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("Figures, table_families.tex, and summary_results.txt generated in", outdir)


if __name__ == "__main__":
    main()
