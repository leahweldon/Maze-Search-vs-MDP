from __future__ import annotations

import pandas as pd


def verify_optimality(input_csv: str):
    df = pd.read_csv(input_csv)

    # Convert numeric fields
    df["path_length"] = pd.to_numeric(df["path_length"], errors="coerce")

    # Only compare successful runs
    df_ok = df[df["success"] == True].copy()  # noqa: E712

    problems = []

    # For each maze, compare BFS vs A* (both heuristics)
    for maze_id, g in df_ok.groupby("maze_id"):
        bfs = g[g["algorithm"] == "bfs"]["path_length"].dropna()
        a1 = g[g["algorithm"] == "astar_manhattan"]["path_length"].dropna()
        a2 = g[g["algorithm"] == "astar_euclidean"]["path_length"].dropna()

        if not bfs.empty:
            bfs_len = bfs.iloc[0]
            if not a1.empty and a1.iloc[0] != bfs_len:
                problems.append((maze_id, "astar_manhattan", a1.iloc[0], bfs_len))
            if not a2.empty and a2.iloc[0] != bfs_len:
                problems.append((maze_id, "astar_euclidean", a2.iloc[0], bfs_len))

        # MDP comparison: VI / PI should match BFS (if both succeed)
        vi = g[g["algorithm"] == "mdp_vi"]["path_length"].dropna()
        pi = g[g["algorithm"] == "mdp_pi"]["path_length"].dropna()

        if not bfs.empty:
            bfs_len = bfs.iloc[0]
            if not vi.empty:
                for v in vi.values:
                    if v != bfs_len:
                        problems.append((maze_id, "mdp_vi", v, bfs_len))
            if not pi.empty:
                for v in pi.values:
                    if v != bfs_len:
                        problems.append((maze_id, "mdp_pi", v, bfs_len))

    if not problems:
        print("✅ Verification passed: all compared path lengths match BFS optimal path length.")
        return

    print("❌ Verification found mismatches (maze_id, algorithm, got, expected_bfs):")
    for row in problems[:50]:
        print(row)

    print(f"\nTotal mismatches: {len(problems)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_csv", required=True)
    args = p.parse_args()

    verify_optimality(args.input_csv)