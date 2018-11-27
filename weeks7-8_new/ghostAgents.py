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
from util import manhattanDistance
import util

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist

class MinimaxGhost(GhostAgent):

    """
      Your minimax agent (question 1)

      useage: python2 pacman.py -p ExpectimaxAgent -l specialNew -g MinimaxGhost -a depth=4
              python2 pacman.py -l specialNew -g MinimaxGhost

    """
    "*** YOUR CODE HERE ***"
    def __init__(self, index, evalFun = 'betterEvaluationFunctionGhost', depth = '2'):
        self.index = index
        self.evaluationFunction = util.lookup(evalFun, globals())
        self.depth = int(depth)

    def getAction(self, state):
        
        def Minimax(state, layer, ghostindex, newindex):
            if state.isWin () or state.isLose():
                return self.evaluationFunction(state)
            if layer > self.depth:
                return self.evaluationFunction(state)
            
            actions = state.getLegalActions(newindex)
            scores=[]
            
            if newindex==0:
                #score=[]
                for i in actions:
                    newstate=state.generateSuccessor(newindex,i)
                    newlayer=layer+1
                    scores.append(Minimax(newstate,newlayer,ghostindex,ghostindex))
            else:
                #scores=[]
                for i in actions:
                    newstate=state.generateSuccessor(newindex,i)
                    scores.append(Minimax(newstate,layer,ghostindex,0))
                    
            return Best(layer,newindex,scores,actions)                    

        def Best(layer,index,scores,actions):
            #Judgement
            if (index!=0)&(layer!=1):
                return min(scores)
            elif (index==0)&(layer!=1):
                return max(scores)
            else:
                bestindex=[]
                for index in range(len(scores)):
                    bestindex.append(index)
            best_action=actions[random.choice(bestindex)]
            return best_action

        startlayer = 1
        ghostindex = self.index
        '''
        actions = state.getLegalActions(ghostindex)
        scores = [getMax(state, startlayer, ghostindex, ghostindex) for iniaction in actions]    
        #print scores
        bestscore = min(scores) #The ghost need the score to be the smallest.
        bestindex = []
        for index in range(len(scores)):
            if scores[index]==bestscore:
                bestindex.append(index)
        best_action=actions[random.choice(bestindex)]
        '''
        output=Minimax(state,startlayer,ghostindex,ghostindex)
        return output


def betterEvaluationFunctionGhost(currentGameState):
    """
        Ghost evaluation function
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()
    eatenGhost = 200
    ghostScore = 0
    ghostWeight = 10
    foodWeight = 10

    for ghost in newGhostStates:
        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        # print "!!!!"
        # print ghostDistance

        if ghostDistance:

            if ghost.scaredTimer:
                update = eatenGhost / float(ghostDistance)
                ghostScore = ghostScore + update
            else:
                update = ghostWeight / ghostDistance
                ghostScore = ghostScore - update


    score = score + ghostScore
    foodDistance = [manhattanDistance(newPos, food) for food in newFood.asList()]

    if foodDistance:
        update = foodWeight / float(min(foodDistance))
        score = score + update

    return score

    # util.raiseNotDefined()


# Abbreviation
ghostEval = betterEvaluationFunctionGhost

