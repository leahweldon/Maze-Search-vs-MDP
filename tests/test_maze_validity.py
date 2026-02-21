from maze.grid import Maze


def test_maze_validation():
    grid = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]

    maze = Maze(grid=grid, start=(0, 0), goal=(2, 2))

    assert maze.in_bounds((1, 2))
    assert maze.is_free((0, 1))
    assert (0, 1) in maze.neighbors_4((0, 0))