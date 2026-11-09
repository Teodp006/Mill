import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.envs.mill.mill_model import MillModel

import random
import math

agent_player = 0
REWARD_WIN = 10
REWARD_CAPTURE = 3



class Node:
    def __init__(self, model, player, parent=None, action=None):
        self.model = model.clone()
        self.player = player
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.legal_moves = self.model.legal_moves(self.player)
        self.is_terminal = self.model.game_over()

    def isFullyExpanded(self):
        return len(self.children) == len(self.legal_moves)

def oponent_player(player):
    return 2 if player == 1 else 1

def expand(node):

    # The possible moves represent the transactions from the parent to it's children for every new "fresh" child so we make sure two child don't represent the same exact space
    posible_moves = [move for move in node.legal_moves if not any(child.action == move for child in node.children)]

    # We select a "not fully random move"
    move = roll_out_move(posible_moves)

    # We create a new model for the child node
    new_model = node.model.clone()

    # We perform the move in the new model
    new_model.make_move(node.player, move)

    # We create the new model
    next_node = Node(model=new_model, player = oponent_player(node.player) , parent=node, action=move)

    # We add the new node to the parents child list
    node.children.append(next_node)

    # Return the choosen child
    return next_node


# Policy to choose next move favorably when there is a possible capture
def roll_out_move(moves):

    # We look if there is any possible move from this stage where we can do a capture
    capturing_moves = [move for move in moves if move[2] > 0]

    # Here when dificulties were implemented we can force MTCS to simulate an scenario where a random move is performed
    if capturing_moves:
        return random.choice(capturing_moves)
    else:
        return random.choice(moves)



def random_playout(node):
    # The actual node copies does a simulation from his current state
    actualmodel = node.model.clone()

    # We start from the player who has the turn
    current_player = node.player

    # We simulate the match until it does not finishes
    while not actualmodel.game_over():
        # We choose a next random move
        move = roll_out_move(actualmodel.legal_moves(current_player))

        # We perform that move in our new model
        actualmodel.make_move(current_player, move) 

        # We switch the turn of the players
        current_player = oponent_player(current_player)

    # We check the end state of the match from the perspective of the root
    match_state = actualmodel.get_phase(agent_player)

    # We return the last node and the state for that node
    return node, -1 if  match_state == "lost" else 1

def select_best_child(node):

    max_uct = -float('inf')
    best_child = None

    for child in node.children:
        if uct_value(child) > max_uct:
            max_uct = uct_value(child)
            best_child = child

    return best_child


def uct_value(node, exploration_weight=math.sqrt(2)):
    # In case the node has never been visited
    if node.visits == 0:
        return float("inf")

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

def montecarlo(iterations, model, player = 0):

    global agent_player
    agent_player = player

    if player == 0:
        raise ValueError("Valor de jugador inválido, introduzca un jugador")

    # Initialice the root node
    root = Node(model=model, player=agent_player)
    

    for _ in range(iterations):
        node = root

        # 1.Selection (Exploration VS Explotation)
        while node.isFullyExpanded() and not node.is_terminal:
            node = select_best_child(node)  # The highest UCT value.

        # 2.Expansion
        if not node.is_terminal:
            node = expand(node)  # Add a randomly chosen child.

        # 3.Simulation
        node, reward = random_playout(node)  # From the root player's point of view.

        # 4.Backpropagation
        while node.parent is not None:
            node.visits += 1
            if oponent_player(node.player) == root.player:  # The move that leads to this node (say the opponent player or the player who is currently playing on this state) it the same that the root player
                node.total_reward += reward
            else:  # The move that lead to this node was made by the opposite player of the root node
                node.total_reward -= reward
            node = node.parent

        # We update also the visits to the root node
        root.visits += 1

    return most_visited_child(root).action
        

