# CS7IS2 — Assignment 1  
## Maze Search vs MDP: Implementation and Performance Evaluation

**Module:** CS7IS2 Artificial Intelligence  
**Assessment Weight:** 35%  
**Type:** Individual Assignment  

---

# Overview

This project implements and experimentally evaluates:

## Search Algorithms
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- A* (Manhattan heuristic)
- A* (Euclidean heuristic)

## Markov Decision Process (MDP) Algorithms
- Value Iteration
- Policy Iteration

All algorithms are applied to mazes of varying sizes and types and compared using multiple performance metrics including:

- Runtime (milliseconds)
- Path length
- Nodes expanded
- Frontier maximum size
- MDP convergence iterations
- Success rate
- Peak memory usage (optional via `tracemalloc`)
- Sensitivity to discount factor (γ)
- Sensitivity to stochastic transition probability (p_intended)
- Sensitivity to maze wall density (random mazes)

This repository contains:

- Full Python implementations of all required algorithms
- Maze generator (perfect + random)
- CLI interface for all functionality
- Automated experiment runner
- YAML-configurable experiment presets
- Plot generation tools
- Raw CSV experimental outputs
- Demo-ready solution exports (PNG + GIF)

This implementation satisfies all requirements of the assignment brief and includes additional experimental extensions (stochastic transitions, density sweeps, large-scale runs).

---

# Repository Structure

```
CS7IS2-ASSIGNMENT1/

config/
    experiments.yaml

demo/

report/
    demo_assets/
    figures/

results/
    mazes/
    raw/
    analysis/
    plots/
    solutions/

src/
    cli.py
    heuristics.py
    metrics.py
    maze/
    solvers/
    experiments/

tests/

requirements.txt
README.md
readme.txt
```

---

# 1️⃣ Setup

## Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest -q
```

Tested with Python 3.11+.

---

# 2️⃣ Maze Generator (Assignment Requirement 1)

The maze generator supports:

- Perfect mazes (tree-structured, DFS backtracker)
- Random mazes with configurable wall density
- Arbitrary dimensions (e.g. 15×15, 31×31, 51×51, 100×100)
- Deterministic reproducibility via seed
- Optional solvability enforcement

No maze sizes are hard-coded.

## Generate Perfect Maze

```bash
python -m src.cli generate \
  --type perfect \
  --width 31 \
  --height 31 \
  --seed 1 \
  --out results/mazes/perfect_31x31_seed1.json
```

## Generate Random Maze

```bash
python -m src.cli generate \
  --type random \
  --width 51 \
  --height 51 \
  --density 0.25 \
  --seed 3 \
  --out results/mazes/random_51x51_seed3.json
```

---

# 3️⃣ Search Algorithms (Assignment Requirement 2)

Implemented in `src/solvers/`.

## DFS
- Stack-based
- Explores deeply before backtracking
- Not guaranteed optimal

## BFS
- Queue-based
- Guarantees shortest path in unweighted grids

## A* (Two Heuristics)
- Manhattan distance
- Euclidean distance

Both heuristics are admissible and consistent in grid-based mazes.

## Solve a Maze

```bash
python -m src.cli solve \
  --maze results/mazes/perfect_31x31_seed1.json \
  --algo bfs
```

Supported algorithms:

- bfs  
- dfs  
- astar_manhattan  
- astar_euclidean  
- mdp_vi  
- mdp_pi  

---

# 4️⃣ MDP Algorithms (Assignment Requirement 3)

Implemented:

- Value Iteration
- Policy Iteration

Configurable parameters:

- Discount factor γ
- Convergence threshold θ
- Intended-action probability (p_intended)

## Deterministic MDP

```bash
python -m src.cli solve \
  --maze results/mazes/perfect_31x31_seed1.json \
  --algo mdp_vi \
  --gamma 0.95 \
  --theta 1e-6
```

## Stochastic MDP Example

```bash
python -m src.cli solve \
  --maze results/mazes/perfect_31x31_seed1.json \
  --algo mdp_vi \
  --gamma 0.95 \
  --theta 1e-6 \
  --p-intended 0.8
```

Where:
- `p_intended = 1.0` → deterministic transitions  
- `p_intended < 1.0` → stochastic transitions  

---

# 5️⃣ Experimental Framework (Assignment Requirement 4)

A complete experiment pipeline is provided.

## Metrics Recorded

- runtime_ms
- path_length
- nodes_expanded
- frontier_max
- success
- iterations (MDP only)
- peak_mem_kb (optional)

Results are saved in:

```
results/raw/
```

---

## Full Experiment (Search + MDP)

```bash
python -m src.cli experiment \
  --type perfect \
  --sizes 15,31,51 \
  --seeds 1,2,3,4,5 \
  --gammas 0.9,0.95,0.99 \
  --theta 1e-6 \
  --out results/raw/final_perfect.csv
```

---

## YAML-Based Experiment Presets (Recommended)

Predefined experiment configurations are stored in:

```
config/experiments.yaml
```

Run a preset:

```bash
python -m src.cli experiment_from_yaml --preset final_perfect_FINAL
```

Preset categories include:

- Final evaluation datasets
- Gamma sensitivity sweeps
- Stochastic transition sweeps
- Density sweeps (random mazes)
- Large-scale search-only experiments (100×100)

---

## Large Maze Search-Only Sweep (e.g. 100×100)

```bash
python -m src.cli experiment_search \
  --type perfect \
  --sizes 100 \
  --seeds 1,2,3 \
  --track-peak-mem \
  --out results/raw/search_only_100.csv
```

---

# 6️⃣ Analysis & Plot Generation

After running experiments:

```bash
python src/experiments/analyze.py \
  --in results/raw/final_perfect.csv \
  --out-dir results/analysis/final_perfect
```

Automatically generated plots include:

- Runtime vs Maze Size
- Nodes Expanded vs Maze Size
- Frontier Size vs Maze Size
- Path Length vs Maze Size
- MDP Runtime vs Gamma
- MDP Iterations vs Gamma
- Success Rate plots
- Search vs MDP comparison plots
- Scaling efficiency plots
- Boxplots for runtime, path length, nodes expanded, frontier size

Outputs are saved in:

```
results/analysis/<experiment_name>/
```

A summary CSV is generated alongside the plots.

---

# 7️⃣ Demo Export (Assignment Requirement 5)

To export PNG and GIF files for demo recording:

```bash
python -m src.cli export_solutions \
  --maze results/mazes/perfect_31x31_seed1.json \
  --out-dir results/solutions/perfect_31x31_seed1 \
  --gamma 0.95 \
  --theta 1e-6 \
  --export-path-gifs \
  --export-frontier-gifs
```

This generates:

- PNG solution images
- Path animation GIFs
- Frontier expansion GIFs

---

# 8️⃣ Testing

Unit tests are included.

Run:

```bash
pytest -q
```

---

# 9️⃣ Submission Contents

The final submission zip file includes:

- Python source code
- Raw CSV experimental outputs
- Generated plots
- PDF performance analysis report
- Demo video (≤ 5 minutes)
- readme.txt (command reference)
- Algorithm implementations appended to report (as required by brief)

---

## License

This project is licensed under the MIT License.
See the LICENSE file for details.