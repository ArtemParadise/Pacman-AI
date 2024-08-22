# ghostAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance, Counter
import util

class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(
                a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(
                pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
                legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist


class BlinkyGhost(GhostAgent):
    "A ghost that directly targets Pac-Man's current position."

    def getDistribution(self, state):
        dist = Counter()
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pacmanPosition = state.getPacmanPosition()
        pos = state.getGhostPosition(self.index)

        distancesToPacman = [manhattanDistance(Actions.getSuccessor(pos, a), pacmanPosition) for a in legalActions]
        bestScore = min(distancesToPacman)
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        for a in bestActions: dist[a] = 1.0 / len(bestActions)
        dist.normalize()
        return dist


class PinkyGhost(GhostAgent):
    "A ghost that targets a position a few steps ahead of Pac-Man."

    def getDistribution(self, state):
        dist = Counter()
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pacmanPosition = state.getPacmanPosition()
        pos = state.getGhostPosition(self.index)

        direction = state.getPacmanState().getDirection()
        vector = Actions.directionToVector(direction, 1)
        targetPosition = (pacmanPosition[0] + vector[0] * 4, pacmanPosition[1] + vector[1] * 4)

        distancesToTarget = [manhattanDistance(Actions.getSuccessor(pos, a), targetPosition) for a in legalActions]
        bestScore = min(distancesToTarget)
        bestActions = [action for action, distance in zip(legalActions, distancesToTarget) if distance == bestScore]

        for a in bestActions: dist[a] = 1.0 / len(bestActions)
        dist.normalize()
        return dist


class InkyGhost(GhostAgent):
    "A ghost that targets a position based on both Blinky and Pac-Man's positions."

    def getDistribution(self, state):
        dist = Counter()
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pacmanPosition = state.getPacmanPosition()
        blinkyPosition = state.getGhostPosition(1)  # Assume Blinky is ghost 1
        pos = state.getGhostPosition(self.index)

        vectorToPacman = (pacmanPosition[0] - blinkyPosition[0], pacmanPosition[1] - blinkyPosition[1])
        targetPosition = (pacmanPosition[0] + vectorToPacman[0], pacmanPosition[1] + vectorToPacman[1])

        distancesToTarget = [manhattanDistance(Actions.getSuccessor(pos, a), targetPosition) for a in legalActions]
        bestScore = min(distancesToTarget)
        bestActions = [action for action, distance in zip(legalActions, distancesToTarget) if distance == bestScore]

        for a in bestActions: dist[a] = 1.0 / len(bestActions)
        dist.normalize()
        return dist


class ClydeGhost(GhostAgent):
    "A ghost that chases Pac-Man when far but retreats when near."

    def getDistribution(self, state):
        dist = Counter()
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pacmanPosition = state.getPacmanPosition()
        pos = state.getGhostPosition(self.index)

        distanceToPacman = manhattanDistance(pos, pacmanPosition)
        if distanceToPacman > 8:
            bestScore = min([manhattanDistance(Actions.getSuccessor(pos, a), pacmanPosition) for a in legalActions])
        else:
            bestScore = max([manhattanDistance(Actions.getSuccessor(pos, a), pacmanPosition) for a in legalActions])

        bestActions = [action for action, distance in zip(legalActions, [
            manhattanDistance(Actions.getSuccessor(pos, a), pacmanPosition) for a in legalActions]) if
                       distance == bestScore]

        for a in bestActions: dist[a] = 1.0 / len(bestActions)
        dist.normalize()
        return dist

