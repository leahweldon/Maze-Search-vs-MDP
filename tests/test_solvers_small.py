from maze.grid import Maze
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.astar import AStarSolver
from solvers.mdp_value_iter import ValueIterationSolver
from solvers.mdp_policy_iter import PolicyIterationSolver


def _simple_maze():
    grid = [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    return Maze(grid=grid, start=(0, 0), goal=(2, 3))


def test_search_algorithms():
    maze = _simple_maze()

    assert BFSSolver().solve(maze).success
    assert DFSSolver().solve(maze).success
    assert AStarSolver("manhattan").solve(maze).success
    assert AStarSolver("euclidean").solve(maze).success


def test_mdp_algorithms():
    maze = _simple_maze()

    vi = ValueIterationSolver(gamma=0.95, theta=1e-6)
    pi = PolicyIterationSolver(gamma=0.95, theta=1e-6)

    assert vi.solve(maze).success
    assert pi.solve(maze).success