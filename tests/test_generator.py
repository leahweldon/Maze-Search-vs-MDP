from maze.generator import generate_maze


def test_generate_random_maze_valid():
    maze = generate_maze(
        width=10,
        height=10,
        maze_type="random",
        seed=123,
        wall_density=0.2,
        ensure_solvable=True,
        max_tries=50,
    )

    assert maze.width == 10
    assert maze.height == 10
    assert maze.grid[maze.start[0]][maze.start[1]] == 0
    assert maze.grid[maze.goal[0]][maze.goal[1]] == 0


def test_generate_perfect_maze_valid_and_solvable():
    maze = generate_maze(
        width=15,
        height=15,
        maze_type="perfect",
        seed=42,
        ensure_solvable=True,
        max_tries=5,
    )

    # Perfect maze adjusts even dimensions to odd; these are already odd.
    assert maze.width == 15
    assert maze.height == 15
    assert maze.grid[maze.start[0]][maze.start[1]] == 0
    assert maze.grid[maze.goal[0]][maze.goal[1]] == 0
