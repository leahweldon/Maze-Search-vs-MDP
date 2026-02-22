from maze.grid import Maze
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.astar import AStarSolver


def _simple_maze():
    grid = [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    return Maze(grid=grid, start=(0, 0), goal=(2, 3))


def _blocked_maze():
    grid = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    return Maze(grid=grid, start=(0, 0), goal=(2, 2))


def test_bfs_finds_shortest_path():
    maze = _simple_maze()
    res = BFSSolver().solve(maze)
    assert res.success is True
    assert res.path_length == 5


def test_dfs_finds_a_path():
    maze = _simple_maze()
    res = DFSSolver().solve(maze)
    assert res.success is True


def test_astar_manhattan_optimal():
    maze = _simple_maze()
    res = AStarSolver("manhattan").solve(maze)
    assert res.success is True
    assert res.path_length == 5


def test_astar_euclidean_optimal():
    maze = _simple_maze()
    res = AStarSolver("euclidean").solve(maze)
    assert res.success is True
    assert res.path_length == 5


def test_no_solution_bfs():
    maze = _blocked_maze()
    res = BFSSolver().solve(maze)
    assert res.success is False


def test_no_solution_dfs():
    maze = _blocked_maze()
    res = DFSSolver().solve(maze)
    assert res.success is False


def test_no_solution_astar():
    maze = _blocked_maze()
    res = AStarSolver("manhattan").solve(maze)
    assert res.success is False