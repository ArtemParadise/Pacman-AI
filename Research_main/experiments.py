import pacman
from pacman import readCommand

# layout names from layouts directory
# -g - Ghost Agent type, f. e. 'DirectionalGhost'
# -n - number of games for 1 run, more info in pacman.py
constant_args = ['-n', '1', '-g', 'DirectionalGhost', '--zoom', 0.5]

SM_MAZES = [
    ['-l', 'smallOptimised'], ### Small 1
    ['-l', 'smallNonOptimised'], ### Small 2 ## redo for AlphaBeta and Expectimax
    ['-l', 'mediumClassic'], ### Medium 1
    ['-l', 'mediumNonOptimized'], ### Medium 2
]

L_MAZES = [
    ['-l', 'originalClassic'], ### Large 1
    ['-l', 'largeNonOptimised'], ### Large 2
]

# Make additionally 4 ghosts 4 goals
XL_MAZES = [
    ['-l', 'xlargeNonOptimised'], ### XLarge 1
]

XL2_MAZES = [
    # ['-l', 'xlargeNonOptimized10w'], ### XLarge 2
    ['-l', 'xlargeNonOptimized15w'], ### XLarge 2
    # ['-l', 'xlargeNonOptimized20w'], ### XLarge 2
]

# Add MCTS later
SM_PACMAN_AGENTS = [
    # ['-p', 'RandomAgent'],
    # ['-p', 'BFSCapsulesSearchAgent'],
    # ['-p', 'AStarCapsulesSearchAgent'],
    # ['-p', 'CapsulesAlphaBetaAgent', '-a', 'evalFn=cbetter,depth=2'],
    # ['-p', 'CapsulesExpectimaxAgent', '-a', 'evalFn=cbetter,depth=2'],
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=10'],
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=30'],
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=50'],
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=75'],
]

# Big-Mazes
L_PACMAN_AGENTS = [
    # ['-p', 'RandomAgent'],
    # ['-p', 'BFSCapsulesSearchAgent'],
    # ['-p', 'AStarCapsulesSearchAgent'],
    # ['-p', 'CapsulesAlphaBetaAgent', '-a', 'evalFn=cbetter,depth=2'],
    # ['-p', 'CapsulesExpectimaxAgent', '-a', 'evalFn=cbetter,depth=1'], ## Redo on XLarge by 50 for 1 time
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=10'],
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=30'],
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=50'],
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=75'],
    # ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=101'],
    # ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=200'],
]

# Big-Mazes
# Try MCTS for 100 simulations
XL_PACMAN_AGENTS = [
    # ['-p', 'RandomAgent'],
    # ['-p', 'CapsulesAlphaBetaAgent', '-a', 'evalFn=cbetter,depth=2'],
    # ['-p', 'CapsulesExpectimaxAgent', '-a', 'evalFn=cbetter,depth=1'],
    ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=50'],
    # ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=101'],
    # ['-p', 'MCTSAgentWithHeuristic', '-a', 'evalFn=mctsbetter,numSimulations=200'],
]

DEFAULTS = ['-g', 'BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost', '-k', '6', '-z', '0.5', '-n', '100'] # Edit n TODO: remove -k param
XL1_DEFAULTS = ['-g', 'BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost', '-k', '6', '-z', '0.5', '-n', '100'] # Edit n
XL2_DEFAULTS = ['-g', 'BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost', '-z', '0.5', '-n', '100'] # Edit n

def run_experiments():
    games_list = []

    # # SM Mazes Games
    # for maze in SM_MAZES:
    #     for pacmanAgent in SM_PACMAN_AGENTS:
    #         params = pacmanAgent + maze + DEFAULTS
    #         print('params -', params)
    #         args = readCommand(params)
    #
    #         games = pacman.runGames(**args)
    #
    #         games_list.append(games)

    # L Mazes Games
    # for maze in L_MAZES:
    #     for pacmanAgent in L_PACMAN_AGENTS:
    #         params = pacmanAgent + maze + DEFAULTS
    #         print('params -', params)
    #         args = readCommand(params)

    #         games = pacman.runGames(**args)

    #         games_list.append(games)

    # XL Mazes Games
    # for maze in XL_MAZES:
    #     for pacmanAgent in XL_PACMAN_AGENTS:
    #         params = pacmanAgent + maze + DEFAULTS
    #         print('params -', params)
    #         args = readCommand(params)

    #         games = pacman.runGames(**args)

    #         games_list.append(games)

    # # XL2 Mazes Games
    for maze in XL2_MAZES:
        for pacmanAgent in XL_PACMAN_AGENTS:
            params = pacmanAgent + maze + XL2_DEFAULTS
            print('params -', params)
            args = readCommand(params)
    
            games = pacman.runGames(**args)
    
            games_list.append(games)

    return games_list

if __name__ == '__main__':
    allGames = run_experiments()

    print("Experiments Finished:", allGames)

# Наброски
# print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        # if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

# Setups
## Random
### Small1 -> python3 pacman.py -l smallOptimised -p RandomAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost
### Small2 -> python3 pacman.py -l smallNonOptimised -p RandomAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost
### Medium1 -> python3 pacman.py -l mediumClassic -p RandomAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost
### Medium2 -> python3 pacman.py -l mediumNonOptimized -p RandomAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost
### Large 1 -> python3 pacman.py -l originalClassic -p RandomAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -z 0.5
### Large 2 -> python3 pacman.py -l largeNonOptimised -p RandomAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -z 0.5
### XLarge -> python3 pacman.py -l xlargeNonOptimised -p RandomAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -z 0.5

# python3 pacman.py -l originalClassicCaps -p SearchAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -a fn=bfs,prob=CapsulesSearchProblem -z 0.5
# python3 pacman.py -l mediumClassicCaps -p SearchAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -a fn=bfs,prob=CapsulesSearchProblem -z 0.5
# python3 pacman.py -l mediumClassicCaps -p AStarCapsulesSearchAgent -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -z 0.5 -n 5
# python3 pacman.py -l originalClassicCaps -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost

# Better depth is 2
# python3 pacman.py -p CapsuleExpectimaxAgent -a evalFn=cbetter,depth=2  -l mediumClassic
# python3 pacman.py -p CapsulesAlphaBetaAgent -a evalFn=cbetter,depth=2 -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost  -l originalClassic
# python3 pacman.py -p CapsulesExpectimaxAgent -a evalFn=cbetter,depth=2 -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost  -l mediumClassic

# MCTS runs
# python3 pacman.py -p MCTSAgentWithHeuristic -a evalFn=mctsbetter -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -l mediumClassic -n 3
# python3 pacman.py -p MCTSAgentWithHeuristic -a evalFn=mctsbetter -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -l originalClassic -n 3


# XL Maze check
# python3 pacman.py -l xlargeNonOptimised -p CapsulesAlphaBetaAgent -a evalFn=cbetter,depth=2 -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -z 0.5
# python3 pacman.py -l xlargeNonOptimised -p CapsulesExpectimaxAgent -a evalFn=cbetter,depth=1 -g BlinkyGhost,PinkyGhost,InkyGhost,ClydeGhost -z 0.5


# Potential performance Diferences
## - depth
## - layouts
## - scared time

# SCARED_TIME
# TIME_PENALTY = 1  # Number of points lost each round

# Further experiments improvements
# Ghosts eaten number
# Goals achieved count