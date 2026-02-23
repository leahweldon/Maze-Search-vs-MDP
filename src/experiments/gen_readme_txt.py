from __future__ import annotations

from pathlib import Path
from datetime import datetime


def generate_readme_txt(
    out_path: str = "readme.txt",
    perfect_csv: str = "results/raw/final_perfect_FINAL.csv",
    random_csv: str = "results/raw/final_random_FINAL.csv",
    perfect_analysis_dir: str = "results/analysis/final_perfect_FINAL",
    random_analysis_dir: str = "results/analysis/final_random_FINAL",
):
    out = Path(out_path)

    content = f"""CS7IS2 Assignment 1 - Maze Search vs MDP
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

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
python -m src.cli experiment --type perfect --sizes 15,31,51 --seeds 1,2,3,4,5 --gammas 0.9,0.95,0.99 --out {perfect_csv}

Random mazes (density recorded in CSV):
python -m src.cli experiment --type random --sizes 15,31,51 --seeds 1,2,3,4,5 --gammas 0.9,0.95,0.99 --density 0.25 --out {random_csv}

------------------------------------------------------------
5) Verify correctness (optimal path consistency)
------------------------------------------------------------
python src/experiments/verify.py --in {perfect_csv}
python src/experiments/verify.py --in {random_csv}

------------------------------------------------------------
6) Analyze results (summary + plots)
------------------------------------------------------------
python src/experiments/analyze.py --in {perfect_csv} --out-dir {perfect_analysis_dir}
python src/experiments/analyze.py --in {random_csv} --out-dir {random_analysis_dir}

(Optional) Pack key figures into report/figures:
python src/experiments/figure_pack.py --perfect-dir {perfect_analysis_dir} --random-dir {random_analysis_dir} --out-dir report/figures

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
"""

    out.write_text(content, encoding="utf-8")
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate submission-ready readme.txt with run commands.")
    p.add_argument("--out", default="readme.txt")
    p.add_argument("--perfect-csv", default="results/raw/final_perfect_FINAL.csv")
    p.add_argument("--random-csv", default="results/raw/final_random_FINAL.csv")
    p.add_argument("--perfect-analysis", default="results/analysis/final_perfect_FINAL")
    p.add_argument("--random-analysis", default="results/analysis/final_random_FINAL")
    args = p.parse_args()

    generate_readme_txt(
        out_path=args.out,
        perfect_csv=args.perfect_csv,
        random_csv=args.random_csv,
        perfect_analysis_dir=args.perfect_analysis,
        random_analysis_dir=args.random_analysis,
    )