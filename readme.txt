CS7IS2 Assignment 1 - Maze Search vs MDP
Generated: 2026-02-23 16:35

This submission contains:
- Maze generator (perfect + random)
- Search solvers: DFS, BFS, A* (Manhattan), A* (Euclidean)
- MDP solvers: Value Iteration, Policy Iteration (deterministic transitions)
- Experiment runner producing CSV results
- Analysis scripts producing summary tables and plots
- Demo animations (path + frontier + live compare) and GIF exporters

------------------------------------------------------------
1) Setup
------------------------------------------------------------
Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

(Optional) Run tests:
pytest -q

------------------------------------------------------------
2) Generate a maze (JSON)
------------------------------------------------------------
Perfect maze (DFS backtracker):

python -m src.cli generate --type perfect --width 31 --height 31 --seed 42 --ensure-solvable --out results/mazes/demo_perfect.json

Random maze (wall density 0.25):

python -m src.cli generate --type random --width 31 --height 31 --seed 42 --density 0.25 --ensure-solvable --out results/mazes/demo_random.json

------------------------------------------------------------
3) Solve a maze (one algorithm)
------------------------------------------------------------
Examples:

python -m src.cli solve --maze results/mazes/demo_perfect.json --algo bfs --png results/plots/demo_bfs.png
python -m src.cli solve --maze results/mazes/demo_perfect.json --algo dfs --png results/plots/demo_dfs.png
python -m src.cli solve --maze results/mazes/demo_perfect.json --algo astar_manhattan --png results/plots/demo_astar_manhattan.png
python -m src.cli solve --maze results/mazes/demo_perfect.json --algo astar_euclidean --png results/plots/demo_astar_euclidean.png

MDP (gamma and theta are configurable):
python -m src.cli solve --maze results/mazes/demo_perfect.json --algo mdp_vi --gamma 0.95 --theta 1e-6 --png results/plots/demo_mdp_vi.png
python -m src.cli solve --maze results/mazes/demo_perfect.json --algo mdp_pi --gamma 0.95 --theta 1e-6 --png results/plots/demo_mdp_pi.png

------------------------------------------------------------
4) Run experiments (writes CSV)
------------------------------------------------------------
Perfect mazes:
python -m src.cli experiment --type perfect --sizes 15,31,51 --seeds 1,2,3,4,5 --gammas 0.9,0.95,0.99 --out results/raw/final_perfect_FINAL.csv

Random mazes (density recorded in CSV):
python -m src.cli experiment --type random --sizes 15,31,51 --seeds 1,2,3,4,5 --gammas 0.9,0.95,0.99 --density 0.25 --out results/raw/final_random_FINAL.csv

------------------------------------------------------------
5) Verify correctness (optimal path consistency)
------------------------------------------------------------
python src/experiments/verify.py --in results/raw/final_perfect_FINAL.csv
python src/experiments/verify.py --in results/raw/final_random_FINAL.csv

------------------------------------------------------------
6) Analyze results (summary + plots)
------------------------------------------------------------
python src/experiments/analyze.py --in results/raw/final_perfect_FINAL.csv --out-dir results/analysis/final_perfect_FINAL
python src/experiments/analyze.py --in results/raw/final_random_FINAL.csv --out-dir results/analysis/final_random_FINAL

(Optional) Pack key figures into report/figures:
python src/experiments/figure_pack.py --perfect-dir results/analysis/final_perfect_FINAL --random-dir results/analysis/final_random_FINAL --out-dir report/figures

------------------------------------------------------------
7) Demo animations
------------------------------------------------------------
Final path animation (GIF):
python -m src.cli animate --maze results/mazes/demo_perfect.json --algo astar_manhattan --out results/plots/path_astar.gif --fps 12

Frontier animation (GIF):
python -m src.cli animate_frontier --maze results/mazes/demo_perfect.json --algo bfs --out results/plots/frontier_bfs.gif --stride 10 --max-frames 400 --fps 12

Live compare (side-by-side window):
python -m src.cli animate_live_compare --maze results/mazes/demo_perfect.json --algo1 bfs --algo2 astar_manhattan --stride 10 --delay 0.02 --final-pause 2.5

Export comparison GIFs for side-by-side inclusion in report:
python -m src.cli animate_compare_gif --maze results/mazes/demo_perfect.json --algo1 bfs --algo2 astar_manhattan --out-dir results/plots/compare_gifs --export-paths

------------------------------------------------------------
End of readme.txt
------------------------------------------------------------
