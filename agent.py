import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.envs.mill.mill_model import MillModel

import random
import math

agent_player = None
REWARD_WIN = 10
REWARD_CAPTURE = 3
EXPLORATION_WEIGHT = math.sqrt(2)

def initialize(player):
    global agent_player
    agent_player = player


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.player = 2 if parent is not None and parent.player == 1 else 1
    
    def isTerminal(self):
        actualmodel = MillModel()
        actualmodel.board = self.state
        return actualmodel.game_over()

    def isFullyExpanded(self):
        actualmodel = MillModel()
        actualmodel.board = self.state
        return len(self.children) == len(actualmodel.legal_moves(self.player))


def expand(node):
    # We create a model from the board representing the state of the node
    modelnode = MillModel()
    modelnode.board = node.state

    # The possible moves represent the transactions from the parent to it's children
    posible_moves = modelnode.legal_moves(node.player)

    # Create a new node for devery transaction
    move = posible_moves[random.randint(0, len(posible_moves) - 1)]

    # If there are more children we check if that move was already done
    if len(node.children) > 0:
        while any(child.action == move for child in node.children):
            move = posible_moves[random.randint(0, len(posible_moves) - 1)]

    # for move in posible_moves:
    #    if not any(child.action == move for child in node.children):
            # Clone the model of the node
    modelnewnode = modelnode.clone()

            # In the model we perform the move to get to a new child
    modelnewnode.make_move(node.player, move)

            # Create a newnode with the current model (represent the new state of the game with action the current move)
    newnode = Node(state=modelnewnode.get_state(), parent=node , action=move)

            # Add node to the list of children
    node.children.append(newnode)

    # Return one random children of the original node
    return newnode

def random_playout(node):
    # The actual node copies does a simulation from his current state
    actualmodel = MillModel()
    actualmodel.board = node.state

    # The end node of the simulation
    leaf = None

    # We simulate the match until it does not finishes
    while not actualmodel.game_over():
        # We choose a next random move
        move = actualmodel.legal_moves(node.player)[random.randint(0, len(actualmodel.legal_moves(node.player)) - 1)]

        # We perform that move in our new model
        actualmodel.make_move(node.player, move) 

        # We create the child that represents that new state
        childmodel = actualmodel.clone()
        child = Node(state=childmodel.get_state(), parent=node, action=move)

        # Then we move to the child node and keep exploring
        node = child

    leaf = node

    return leaf, -1 if actualmodel.get_phase(node.player) == "lost" else 1

def select_best_child(node):

    max_uct = -float('inf')
    best_child = None

    for child in node.children:
        if uct_value(child) > max_uct:
            max_uct = uct_value(child)
            best_child = child

    return best_child


def uct_value(node, exploration_weight=EXPLORATION_WEIGHT):
    if node.visits == 0:
        return 0

    exploration = exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
    exploitation = node.total_reward / node.visits

    return exploration + exploitation

def most_visited_child(node):
    max_visited = -float('inf')
    best_child = None
    for child in node.children:
        if child.visits > max_visited:
            max_visited = child.visits
            best_child = child
    return best_child 

def montecarlo(iterations, model):
    clone_model = model.clone()

    root = Node(state=clone_model.get_state())
    root.player = agent_player

    for _ in range(iterations):
        node = root

        # 1.Selection (Exploration VS Explotation)
        while node.isFullyExpanded() and not node.isTerminal():
            node = select_best_child(node)  # The highest UCT value.

        # 2.Expansion
        if not node.isTerminal():
            node = expand(node)  # Add a randomly chosen child.

        # 3.Simulation
        node, reward = random_playout(node)  # From the root player's point of view.

        # 4.Backpropagation
        while node is not None:
            node.visits += 1
            if node.player == root.player:  # The move into node was made by the root player.
                node.total_reward += reward
            else:  # The opponent's move: negate the reward.
                node.total_reward -= reward
            node = node.parent

    return most_visited_child(root).action
        

