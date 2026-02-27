CS7IS2 — Assignment 1
Maze Search vs MDP Solver and Performance Evaluation
------------------------------------------------------

This submission implements:

• Maze generator (perfect and random mazes)
• Search algorithms:
    - Depth-First Search (DFS)
    - Breadth-First Search (BFS)
    - A* (Manhattan heuristic)
    - A* (Euclidean heuristic)
• MDP algorithms:
    - Value Iteration
    - Policy Iteration
• Deterministic and stochastic MDP transitions
• Full experimental framework producing CSV output
• YAML-configurable experiment presets
• Automated analysis and plotting scripts
• Demo animations (path + frontier + side-by-side compare)
• Large-maze search-only mode (e.g. 100x100)
• Optional peak memory tracking (tracemalloc)

All maze sizes are configurable. No maze sizes are hard-coded.

------------------------------------------------------------
1) Setup
------------------------------------------------------------

Create virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

(Optional) Run tests:

pytest -q

------------------------------------------------------------
2) Generate a Maze (JSON Output)
------------------------------------------------------------

Perfect maze (DFS backtracker):

python -m src.cli generate \
  --type perfect \
  --width 31 \
  --height 31 \
  --seed 42 \
  --ensure-solvable \
  --out results/mazes/demo_perfect.json

Random maze (density = 0.25):

python -m src.cli generate \
  --type random \
  --width 31 \
  --height 31 \
  --seed 42 \
  --density 0.25 \
  --ensure-solvable \
  --out results/mazes/demo_random.json

------------------------------------------------------------
3) Solve a Maze (Single Algorithm)
------------------------------------------------------------

Search algorithms:

python -m src.cli solve --maze results/mazes/demo_perfect.json --algo dfs
python -m src.cli solve --maze results/mazes/demo_perfect.json --algo bfs
python -m src.cli solve --maze results/mazes/demo_perfect.json --algo astar_manhattan
python -m src.cli solve --maze results/mazes/demo_perfect.json --algo astar_euclidean

Export solution image:

python -m src.cli solve \
  --maze results/mazes/demo_perfect.json \
  --algo bfs \
  --png results/plots/demo_bfs.png

------------------------------------------------------------
MDP Algorithms (Deterministic)
------------------------------------------------------------

python -m src.cli solve \
  --maze results/mazes/demo_perfect.json \
  --algo mdp_vi \
  --gamma 0.95 \
  --theta 1e-6

python -m src.cli solve \
  --maze results/mazes/demo_perfect.json \
  --algo mdp_pi \
  --gamma 0.95 \
  --theta 1e-6

------------------------------------------------------------
MDP Algorithms (Stochastic Transitions)
------------------------------------------------------------

p_intended = probability that chosen action executes as intended.
Remaining probability is distributed across other valid moves.

Example with p_intended = 0.8:

python -m src.cli solve \
  --maze results/mazes/demo_perfect.json \
  --algo mdp_vi \
  --gamma 0.95 \
  --theta 1e-6 \
  --p-intended 0.8

------------------------------------------------------------
4) Full Experiment Runs (Search + MDP)
------------------------------------------------------------

Perfect mazes:

python -m src.cli experiment \
  --type perfect \
  --sizes 15,31,51 \
  --seeds 1,2,3,4,5 \
  --gammas 0.9,0.95,0.99 \
  --theta 1e-6 \
  --out results/raw/final_perfect_FINAL.csv

Random mazes (density recorded in CSV):

python -m src.cli experiment \
  --type random \
  --sizes 15,31,51 \
  --seeds 1,2,3,4,5 \
  --gammas 0.9,0.95,0.99 \
  --density 0.25 \
  --theta 1e-6 \
  --out results/raw/final_random_FINAL.csv

------------------------------------------------------------
5) YAML Preset Experiments (Recommended)
------------------------------------------------------------

Presets are defined in:

config/experiments.yaml

Run a preset:

python -m src.cli experiment_from_yaml --preset final_perfect_FINAL

Examples of presets included:
• Final evaluation runs
• Gamma sensitivity sweep
• Stochastic transition sweep
• Density sweep (random mazes)
• Large-scale search-only experiments

------------------------------------------------------------
6) Large Maze Mode (Search Only — e.g. 100x100)
------------------------------------------------------------

MDP becomes computationally expensive at very large sizes.
Search-only mode is provided.

Perfect 100x100:

python -m src.cli experiment_search \
  --type perfect \
  --sizes 100 \
  --seeds 1,2,3 \
  --track-peak-mem \
  --out results/raw/search_only_perfect_100.csv

Random 100x100:

python -m src.cli experiment_search \
  --type random \
  --sizes 100 \
  --seeds 1,2,3 \
  --density 0.25 \
  --track-peak-mem \
  --out results/raw/search_only_random_100.csv

------------------------------------------------------------
7) Verify Correctness (Optimality Checks)
------------------------------------------------------------

python src/experiments/verify.py \
  --in results/raw/final_perfect_FINAL.csv

python src/experiments/verify.py \
  --in results/raw/final_random_FINAL.csv

------------------------------------------------------------
8) Generate Analysis (Summary + Plots)
------------------------------------------------------------

Perfect:

python src/experiments/analyze.py \
  --in results/raw/final_perfect_FINAL.csv \
  --out-dir results/analysis/final_perfect_FINAL

Random:

python src/experiments/analyze.py \
  --in results/raw/final_random_FINAL.csv \
  --out-dir results/analysis/final_random_FINAL

This produces:

• Summary CSV
• Runtime vs size plots
• Nodes expanded vs size plots
• Frontier size vs size plots
• Path length plots
• Efficiency plots
• Success rate plots
• MDP gamma comparison plots
• Stochastic sensitivity plots
• Search vs MDP comparison plots
• Boxplots for multiple metrics

------------------------------------------------------------
9) Optional Figure Pack for Report
------------------------------------------------------------

python src/experiments/figure_pack.py \
  --perfect-dir results/analysis/final_perfect_FINAL \
  --random-dir results/analysis/final_random_FINAL \
  --out-dir report/figures

------------------------------------------------------------
10) Demo Animations
------------------------------------------------------------

Path animation (GIF):

python -m src.cli animate \
  --maze results/mazes/demo_perfect.json \
  --algo astar_manhattan \
  --out results/plots/path_astar.gif \
  --fps 12

Frontier animation (GIF):

python -m src.cli animate_frontier \
  --maze results/mazes/demo_perfect.json \
  --algo bfs \
  --out results/plots/frontier_bfs.gif \
  --stride 10 \
  --max-frames 400 \
  --fps 12

Live side-by-side compare:

python -m src.cli animate_live_compare \
  --maze results/mazes/demo_perfect.json \
  --algo1 bfs \
  --algo2 astar_manhattan \
  --stride 10 \
  --delay 0.02 \
  --final-pause 2.5

Export side-by-side GIFs:

python -m src.cli animate_compare_gif \
  --maze results/mazes/demo_perfect.json \
  --algo1 bfs \
  --algo2 astar_manhattan \
  --out-dir results/plots/compare_gifs \
  --export-paths

------------------------------------------------------------
End of readme.txt
------------------------------------------------------------