from maze.grid import Maze
from solvers.bfs import BFSSolver


def test_bfs_finds_shortest_path_in_simple_maze():
    grid = [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    maze = Maze(grid=grid, start=(0, 0), goal=(2, 3))

    res = BFSSolver().solve(maze)

    assert res.success is True
    assert res.path is not None
    assert res.path[0] == (0, 0)
    assert res.path[-1] == (2, 3)
    # shortest moves here should be 5:
    # (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)
    assert res.path_length == 5