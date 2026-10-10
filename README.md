# Collaborative Forgetting — Anonymous Artifact

This repository contains the code and experimental results accompanying the anonymous submission:

**Forgetting in the Dataflow: Modeling and Propagating Deletion in Multi-Party Provenance Graphs**

The artifact implements the Collaborative Forgetting Graph (CFG), CascadeDelete,
GreedyRepair, k-step Downstream-Aware Repair (DA-k), and the exact
integer-programming formulation of MCCF solved with OR-Tools CP-SAT.

## Requirements

- Python 3.10+
- matplotlib 3.7+
- ortools 9.x (exact CP-SAT solver)

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Repository structure

- `src/collaborative_forgetting_experiments.py`:
  CFG generator and repair algorithms.
- `scripts/mccf_ilp.py`:
  exact integer-programming formulation of MCCF (OR-Tools CP-SAT).
- `scripts/run_ilp.py`, `scripts/run_ilp_scalability.py`, `scripts/run_ablation.py`:
  exact-optimum, exact-solver scalability, and action-ablation campaigns.
- `scripts/generate_paper_outputs.py`:
  generates figures and aggregate tables from experiment CSVs.
- `results/`:
  CSV files corresponding to the campaigns reported in the paper.
- `figures/`:
  paper figures generated from those CSV files.

## Retention metric

The artifact uses the retention definition reported in the paper:

\[
\mathrm{Retention}(\pi)
=
\frac{
|\{v \in V_{\mathcal R}\setminus S :
\mathrm{active}_{\pi}(v)\}|
}{
|V_{\mathcal R}\setminus S|
}.
\]

Artifacts explicitly requested for forgetting are excluded from both numerator
and denominator.

## Reproducing the experiments

### Gap to the proven optimum (RQ1)

Solves the 180 adversarial instances of RQ2 exactly and runs all heuristics:

```bash
python scripts/run_ilp.py              # -> results/results_ilp.csv
```

### Exact solver at scale (RQ4)

```bash
python scripts/run_ilp_scalability.py  # -> results/results_ilp_scalability.csv
```

Graph generation for 50,000 and 100,000 nodes dominates the running time
(several hours in total).

### Action ablation

```bash
python scripts/run_ablation.py         # -> results/results_ablation.csv
```

### Look-ahead depth experiment

```bash
python src/collaborative_forgetting_experiments.py \
  --n 50,100,200 \
  --actors 3,5 \
  --families ADVERSARIAL \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --trap-depths 1,2,3 \
  --lookahead-depths 1,2,3 \
  --lambdas 1 \
  --exact-limit 0 \
  --out results/results_depth_k.csv
```

### Lambda sensitivity

```bash
python src/collaborative_forgetting_experiments.py \
  --n 100 \
  --actors 5 \
  --families ADVERSARIAL \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --trap-depths 2 \
  --lookahead-depths 1,2,3 \
  --lambdas 0,0.25,0.5,1,2,4 \
  --exact-limit 0 \
  --out results/results_lambda.csv
```

### Comparison across graph families

```bash
python src/collaborative_forgetting_experiments.py \
  --n 50,100,200 \
  --actors 3,5 \
  --families LOCAL,SHARED,ADVERSARIAL \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --trap-depths 1 \
  --lookahead-depths 1,3 \
  --lambdas 1 \
  --exact-limit 0 \
  --out results/results_families.csv
```

### Controlled scalability experiment

```bash
python src/collaborative_forgetting_experiments.py \
  --n 1000,5000,10000,50000,100000 \
  --actors 5 \
  --families LOCAL,SHARED,ADVERSARIAL \
  --seeds 1,2,3,4,5 \
  --trap-depths 1 \
  --lookahead-depths 1 \
  --lambdas 1 \
  --affected-fractions 0.1,0.25,0.5 \
  --controlled-affected \
  --exact-limit 0 \
  --out results/results_scalability.csv
```

The largest scalability configurations can take substantially longer than the other experiments.

## Generating the paper figures

Run from the repository root:

```bash
python scripts/generate_paper_outputs.py --input-dir results --output-dir figures
```

This produces the three figures, `table_families.tex`, and
`summary_results.txt`.

The plotting script uses embedded TrueType fonts in PDF output:

```python
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
```

## Main expected results

- Against proven optima, GreedyRepair is 110-141% more expensive on adversarial
  instances; DA-k is within 1% on average whenever k >= trap depth.
- Disabling recompute raises the optimal cost by 15% (Shared) and 156%
  (Adversarial); disabling unlearn by 1% and 4%.
- On depth-one adversarial instances, DA-1 reduces mean repair cost from
  126.06 to 65.59.
- On depth-three traps, mean repair cost decreases from 151.24 (Greedy)
  to 122.76, 87.50, and 67.67 for DA-1, DA-2, and DA-3.
- Retention on depth-three traps increases from 10.6% to 28.1%, 50.8%,
  and 68.8%.
- In the lambda sensitivity experiment, DA-2 and DA-3 reach mean cost
  66.33 at lambda=0.25 and 65.89 from lambda=0.5 onward.

## Anonymity

This repository intentionally contains no author-identifying information during
the double-anonymous review process.
