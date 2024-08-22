import pacman
from pacman import readCommand

# constant_args = ['-l', 'originalClassic', '--zoom', 0.5]
# constant_args = ['-l', 'bigMaze', '-p', 'GoWestAgent' '--zoom', 0.5]
# constant_args = []
n_experiments = '5'

# Add Param for eatable ghosts after eating capsule

# layout names from layouts directory
# mazes = ['mediumClassicCaps', 'originalClassicCaps', 'powerClassic', 'bigSafeSearch']
mazes = ['mediumClassicCaps', 'originalClassicCaps', 'bigSearchCaps', 'bigMazeCaps']
# -g - Ghost Agent type, f. e. 'DirectionalGhost'
# -n - number of games for 1 run, more info in pacman.py
constant_args = ['-n', '1', '-g', 'DirectionalGhost', '--zoom', 0.5]
def run_mc_experiments():
    results = []
    for distance in range(1, 5):
        # -p - Pacman agents type from *Agents files, more info in pacman.py
        dynamic_params = ['-p', 'MonteCarloPacmanAgent', f'-a', f'optimal_distance={distance}']
        all_params = constant_args + dynamic_params

        args = readCommand(all_params)
        games = pacman.runGames(**args)

        score = sum([game.state.getScore() for game in games]) / len(games)
        time_in_game = len(games)
        results.append((distance, score, time_in_game))
    return results

def run_experiments(agentType):
    results_list = []

    for mazeType in mazes:
        dynamic_params = ['-p', agentType, '-l', mazeType]
        all_params = constant_args + dynamic_params
        args = readCommand(all_params)
        games = pacman.runGames(**args)

        results_list.append({mazeType: [game.state.getScore() for game in games]})

    return results_list

if __name__ == '__main__':
    random_results = run_experiments('RandomPacmanAgent')
    go_lt_results = run_experiments('LeftTurnAgent')
    monte_carlo_results = run_mc_experiments()
    # go_sa_results = run_experiments('SearchAgent')
    # go_psp_results = run_experiments('PositionSearchProblem')

    print("Random Results:", random_results)
    print("Left Turn Results:", go_lt_results)
    print("Monte Carlo Results:", monte_carlo_results)
    # print("SearchAgent Results:", go_sa_results)
    # print("PositionSearchProblem Results:", go_psp_results)


# Наброски
# print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        # if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)
# Setups
# python3 pacman.py -l originalClassicCaps -p SearchAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -a fn=bfs,prob=CapsulesSearchProblem -z 0.5
# python3 pacman.py -l mediumClassicCaps -p SearchAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -a fn=bfs,prob=CapsulesSearchProblem -z 0.5
# python3 pacman.py -l mediumClassicCaps -p AStarCapsulesSearchAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -z 0.5 -n 5
# python3 pacman.py -l originalClassicCaps -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost

# SCARED_TIME
# TIME_PENALTY = 1  # Number of points lost each round
