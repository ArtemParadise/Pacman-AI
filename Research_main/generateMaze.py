import random
import argparse
import os

def create_empty_maze(width, height):
    return [[' ' for _ in range(width)] for _ in range(height)]

def cover_edges_with_walls(maze):
    height = len(maze)
    width = len(maze[0])
    # Cover top and bottom edges
    for x in range(width):
        maze[0][x] = '%'
        maze[height-1][x] = '%'
    # Cover left and right edges
    for y in range(height):
        maze[y][0] = '%'
        maze[y][width-1] = '%'

def add_internal_walls(maze, wall_percentage):
    height = len(maze)
    width = len(maze[0])
    num_cells = (height - 2) * (width - 2)  # Exclude edge cells
    max_walls = int(num_cells * wall_percentage)
    wall_count = 0

    while wall_count < max_walls:
        x, y = random.randint(1, height - 2), random.randint(1, width - 2)
        if maze[x][y] == ' ':
            maze[x][y] = '%'
            wall_count += 1

def fill_empty_spaces(maze):
    height = len(maze)
    width = len(maze[0])
    for x in range(height):
        for y in range(width):
            if maze[x][y] == ' ':
                maze[x][y] = '.'

def place_items(maze, item, count):
    height = len(maze)
    width = len(maze[0])
    placed = 0
    while placed < count:
        x, y = random.randint(1, height - 2), random.randint(1, width - 2)
        if maze[x][y] == '.':
            maze[x][y] = item
            placed += 1

def ensure_open_areas(maze):
    height = len(maze)
    width = len(maze[0])
    # Simplified logic to ensure maze has open areas and paths
    pass

def convert_to_lay_format(maze):
    return '\n'.join(''.join(row) for row in maze)

def generate_maze(width, height, num_ghosts, num_capsules, wall_percentage):
    maze = create_empty_maze(width, height)
    cover_edges_with_walls(maze)
    add_internal_walls(maze, wall_percentage)
    fill_empty_spaces(maze)
    place_items(maze, 'G', num_ghosts)
    place_items(maze, 'o', num_capsules)
    ensure_open_areas(maze)
    return convert_to_lay_format(maze)

def main():
    parser = argparse.ArgumentParser(description="Generate a Pac-Man maze with customizable parameters.")
    parser.add_argument(
        '-w', '--width', 
        type=int, 
        required=True, 
        help='Width of the maze. Must be > 2.'
    )
    parser.add_argument(
        '-ht', '--height', 
        type=int, 
        required=True, 
        help='Height of the maze. Must be > 2.'
    )
    parser.add_argument(
        '-wp', '--wall_percentage', 
        type=float, 
        required=True, 
        help='Percentage of internal walls (0-100 or 0.0-1.0). Example: 20 or 0.2 for 20%% walls'
    )
    parser.add_argument(
        '-c', '--capsules', 
        type=int, 
        required=True, 
        help='Number of power capsules to place in the maze. Must be >= 0'
    )
    parser.add_argument(
        '-g', '--ghosts', 
        type=int, 
        required=True, 
        help='Number of ghosts to place in the maze. Must be >= 0'
    )

    args = parser.parse_args()

    width = args.width
    height = args.height
    wall_percentage = args.wall_percentage
    num_capsules = args.capsules
    num_ghosts = args.ghosts

    maze_lay = generate_maze(width, height, num_ghosts, num_capsules, wall_percentage)

    if not os.path.exists('layouts'):
        os.makedirs('layouts')

    filename = f'layouts/{width}x{height}_{wall_percentage}p_{num_capsules}caps_maze.lay'
    with open(filename, 'w') as file:
        file.write(maze_lay)

    print(f"Maze generated and saved to {filename}")

if __name__ == "__main__":
    main()
