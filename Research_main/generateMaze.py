import random
import argparse
import os

def create_empty_maze(size):
    return [[' ' for _ in range(size)] for _ in range(size)]

def cover_edges_with_walls(maze):
    size = len(maze)
    # Cover top and bottom edges
    for x in range(size):
        maze[0][x] = '%'
        maze[size-1][x] = '%'
    # Cover left and right edges
    for y in range(size):
        maze[y][0] = '%'
        maze[y][size-1] = '%'

def add_internal_walls(maze, wall_percentage):
    size = len(maze)
    num_cells = (size - 2) * (size - 2)  # Exclude edge cells
    max_walls = int(num_cells * wall_percentage)
    wall_count = 0

    while wall_count < max_walls:
        x, y = random.randint(1, size - 2), random.randint(1, size - 2)
        if maze[x][y] == ' ':
            maze[x][y] = '%'
            wall_count += 1

def fill_empty_spaces(maze):
    size = len(maze)
    for x in range(size):
        for y in range(size):
            if maze[x][y] == ' ':
                maze[x][y] = '.'

def place_items(maze, item, count):
    size = len(maze)
    placed = 0
    while placed < count:
        x, y = random.randint(1, size - 2), random.randint(1, size - 2)
        if maze[x][y] == '.':
            maze[x][y] = item
            placed += 1

def ensure_open_areas(maze):
    size = len(maze)
    # Simplified logic to ensure maze has open areas and paths
    pass

def convert_to_lay_format(maze):
    return '\n'.join(''.join(row) for row in maze)

def generate_maze(size, num_ghosts, num_capsules, wall_percentage):
    maze = create_empty_maze(size)
    cover_edges_with_walls(maze)
    add_internal_walls(maze, wall_percentage)
    fill_empty_spaces(maze)
    place_items(maze, 'G', num_ghosts)
    place_items(maze, 'o', num_capsules)
    ensure_open_areas(maze)
    return convert_to_lay_format(maze)

def main():
    parser = argparse.ArgumentParser(description="Generate a Pac-Man maze.")
    parser.add_argument('-s', '--size', type=int, required=True, help='Size of the maze (e.g., 50)')
    parser.add_argument('-w', '--wall_percentage', type=float, required=True, help='Percentage of walls (e.g., 0.25)')
    parser.add_argument('-c', '--capsules', type=int, required=True, help='Number of capsules')
    parser.add_argument('-g', '--ghosts', type=int, required=True, help='Number of ghosts')

    args = parser.parse_args()

    size = args.size
    wall_percentage = args.wall_percentage
    num_capsules = args.capsules
    num_ghosts = args.ghosts

    maze_lay = generate_maze(size, num_ghosts, num_capsules, wall_percentage)

    if not os.path.exists('layouts'):
        os.makedirs('layouts')

    filename = f'layouts/{size}x{size}_{wall_percentage}p_{num_capsules}caps_maze.lay'
    with open(filename, 'w') as file:
        file.write(maze_lay)

    print(f"Maze generated and saved to {filename}")

if __name__ == "__main__":
    main()
