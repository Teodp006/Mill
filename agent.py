import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.envs.mill.mill_model import MillModel

import random
import math

agent_player = None
REWARD_WIN = 10
REWARD_CAPTURE = 3
REWARD_MILL = 4
REWARD_BLOCK = 3
REWARD_THREAT_CAPTURE = 2

def initialize(player):
    global agent_player
    agent_player = player


def _count_threats(model, player):
    board = model._board
    threats = 0
    for mill in MillModel.mills:
        values = [board[position] for position in mill]
        if values.count(player) == 2 and values.count(0) == 1:
            threats += 1
    return threats


def _forms_mill(model, player, position):
    board = model._board
    return any(
        position in mill and all(board[index] == player for index in mill)
        for mill in MillModel.mills
    )


def _threats_containing(model, player, position):
    board = model._board
    return sum(
        position in mill
        and [board[index] for index in mill].count(player) == 2
        and [board[index] for index in mill].count(0) == 1
        for mill in MillModel.mills
    )


def _tactical_score(model, player, move):
    src, dst, take = move
    opponent = 2 if player == 1 else 1
    board = model._board
    opponent_threats_before = _count_threats(model, opponent)
    captured_threats = (
        _threats_containing(model, opponent, take) if take else 0
    )
    original_src = board[src]
    original_dst = board[dst]
    original_take = board[take] if take else 0

    if src:
        board[src] = 0
    board[dst] = player
    if take:
        board[take] = 0

    score = 0
    if _forms_mill(model, player, dst):
        score += REWARD_MILL

    blocked_threats = opponent_threats_before - _count_threats(model, opponent)
    score += blocked_threats * REWARD_BLOCK

    if take:
        score += REWARD_CAPTURE
        score += captured_threats * REWARD_THREAT_CAPTURE

    board[src] = original_src
    board[dst] = original_dst
    if take:
        board[take] = original_take

    return score


def _prioritized_choice(model, player, moves):
    scores = [_tactical_score(model, player, move) for move in moves]
    weights = [1 + max(score, 0) for score in scores]
    return random.choices(moves, weights=weights, k=1)[0]


def _move_forms_mill(model, player, move):
    candidate = model.clone()
    candidate.make_move(player, move)
    return _forms_mill(candidate, player, move[1])


def _has_immediate_mill(model, player):
    return any(
        _move_forms_mill(model, player, move)
        for move in model.legal_moves(player)
    )


def _tactical_root_moves(model, player, moves):
    mill_moves = [
        move for move in moves
        if _move_forms_mill(model, player, move)
    ]
    if mill_moves:
        return mill_moves

    opponent = 2 if player == 1 else 1
    if not _has_immediate_mill(model, opponent):
        return moves

    blocking_moves = []
    for move in moves:
        candidate = model.clone()
        candidate.make_move(player, move)
        if not _has_immediate_mill(candidate, opponent):
            blocking_moves.append(move)

    return blocking_moves or moves


class Node:
    def __init__(self, model, parent=None, action=None, allowed_actions=None):
        self.model = model
        self.parent = parent
        self.action = action
        self.allowed_actions = allowed_actions
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.player = 2 if parent is not None and parent.player == 1 else 1
        self._legal_moves = None
        self.expanded_actions = set()
    
    def legal_moves(self):
        if self._legal_moves is None:
            self._legal_moves = self.model.legal_moves(self.player)
            if self.allowed_actions is not None:
                allowed = {tuple(move) for move in self.allowed_actions}
                self._legal_moves = [
                    move for move in self._legal_moves
                    if tuple(move) in allowed
                ]
        return self._legal_moves

    def is_terminal(self):
        return self.model.game_over()

    def is_fully_expanded(self):
        return len(self.expanded_actions) == len(self.legal_moves())


def expand(node):
    possible_moves = node.legal_moves()
    unexpanded_moves = [
        move for move in possible_moves
        if tuple(move) not in node.expanded_actions
    ]

    move = _prioritized_choice(node.model, node.player, unexpanded_moves)
    node.expanded_actions.add(tuple(move))

    modelnewnode = node.model.clone()
    modelnewnode.make_move(node.player, move)
    newnode = Node(model=modelnewnode, parent=node, action=move)
    node.children.append(newnode)
    return newnode

def random_playout(node, root_player):
    actualmodel = node.model.clone()
    player = node.player
    while not actualmodel.game_over():
        moves = actualmodel.legal_moves(player)
        if not moves:
            break
        actualmodel.make_move(
            player,
            _prioritized_choice(actualmodel, player, moves),
        )
        player = 2 if player == 1 else 1

    winner = 2 if player == 1 else 1
    reward = 1 if winner == root_player else -1
    return node, reward

def select_best_child(node):

    max_uct = -float('inf')
    best_child = None

    for child in node.children:
        if uct_value(child) > max_uct:
            max_uct = uct_value(child)
            best_child = child

    return best_child


def uct_value(node, exploration_weight=math.sqrt(2)):
    if node.visits == 0:
        return float('inf')

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

    root_moves = clone_model.legal_moves(agent_player)
    root_moves = _tactical_root_moves(clone_model, agent_player, root_moves)
    root = Node(model=clone_model, allowed_actions=root_moves)
    root.player = agent_player

    for _ in range(iterations):
        node = root

        # 1.Selection (Exploration VS Explotation)
        while node.is_fully_expanded() and not node.is_terminal():
            node = select_best_child(node)  # The highest UCT value.

        # 2.Expansion
        if not node.is_terminal():
            node = expand(node)  # Add a randomly chosen child.

        # 3.Simulation
        node, reward = random_playout(node, root.player)

        # 4.Backpropagation
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    return most_visited_child(root).action
        
