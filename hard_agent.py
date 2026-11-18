import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.envs.mill.mill_model import MillModel

import random
import math

agent_player = None

# Scores for useful moves.
REWARD_WIN = 10
REWARD_CAPTURE = 3
REWARD_MILL = 4
REWARD_BLOCK = 3
REWARD_THREAT_CAPTURE = 2


# Set the player controlled by the agent.
def initialize(player):
    global agent_player
    agent_player = player


# Count the player's open mills.
def _count_threats(model, player):
    board = model._board
    threats = 0
    for mill in MillModel.mills:
        values = [board[position] for position in mill]
        if values.count(player) == 2 and values.count(0) == 1:
            threats += 1
    return threats


# Check whether a move completes a mill.
def _forms_mill(model, player, position):
    board = model._board
    return any(
        position in mill and all(board[index] == player for index in mill)
        for mill in MillModel.mills
    )


# Count open mills that include a position.
def _threats_containing(model, player, position):
    board = model._board
    return sum(
        position in mill
        and [board[index] for index in mill].count(player) == 2
        and [board[index] for index in mill].count(0) == 1
        for mill in MillModel.mills
    )


# Score a move using immediate tactical benefits.
def _tactical_score(model, player, move):
    src, dst, take = move
    opponent = 2 if player == 1 else 1
    board = model._board

    # Count opponent threats before applying the move.
    opponent_threats_before = _count_threats(model, opponent)
    captured_threats = (
        _threats_containing(model, opponent, take) if take else 0
    )

    # Save the affected positions before simulating the move.
    original_src = board[src]
    original_dst = board[dst]
    original_take = board[take] if take else 0

    # Apply the move temporarily.
    if src:
        board[src] = 0
    board[dst] = player
    if take:
        board[take] = 0

    score = 0

    # Reward moves that create a mill.
    if _forms_mill(model, player, dst):
        score += REWARD_MILL

    # Reward moves that remove opponent threats.
    blocked_threats = opponent_threats_before - _count_threats(model, opponent)
    score += blocked_threats * REWARD_BLOCK

    # Reward captures and captures that break threats.
    if take:
        score += REWARD_CAPTURE
        score += captured_threats * REWARD_THREAT_CAPTURE

    # Restore the board after scoring the move.
    board[src] = original_src
    board[dst] = original_dst
    if take:
        board[take] = original_take

    return score


# Choose a move with higher-scoring moves weighted more heavily.
def _prioritized_choice(model, player, moves):
    scores = [_tactical_score(model, player, move) for move in moves]
    weights = [1 + max(score, 0) for score in scores]
    return random.choices(moves, weights=weights, k=1)[0]


# Check whether a move completes a mill.
def _move_forms_mill(model, player, move):
    candidate = model.clone()
    candidate.make_move(player, move)
    return _forms_mill(candidate, player, move[1])


# Check whether the player can complete a mill.
def _has_immediate_mill(model, player):
    return any(
        _move_forms_mill(model, player, move)
        for move in model.legal_moves(player)
    )


# Keep winning moves and block immediate opponent mills.
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
    # Store one position in the search tree.
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
    
    # Return legal moves that are allowed for this node.
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

    # Check whether the game has ended.
    def is_terminal(self):
        return self.model.game_over()

    # Check whether every legal move has been explored.
    def is_fully_expanded(self):
        return len(self.expanded_actions) == len(self.legal_moves())


# Add one unexplored move to the search tree.
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


# Play a game from the node and return the result.
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


# Select the child with the highest UCT value.
def select_best_child(node):

    max_uct = -float('inf')
    best_child = None

    for child in node.children:
        if uct_value(child) > max_uct:
            max_uct = uct_value(child)
            best_child = child

    return best_child


# Balance exploration and exploitation.
def uct_value(node, exploration_weight=math.sqrt(2)):
    if node.visits == 0:
        return float('inf')

    exploration = exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
    exploitation = node.total_reward / node.visits

    return exploration + exploitation


# Select the child visited most often.
def most_visited_child(node):
    max_visited = -float('inf')
    best_child = None
    for child in node.children:
        if child.visits > max_visited:
            max_visited = child.visits
            best_child = child
    return best_child 


# Run Monte Carlo Tree Search.
def montecarlo(iterations, model):
    clone_model = model.clone()

    root_moves = clone_model.legal_moves(agent_player)
    root_moves = _tactical_root_moves(clone_model, agent_player, root_moves)
    root = Node(model=clone_model, allowed_actions=root_moves)
    root.player = agent_player

    for _ in range(iterations):
        node = root

        # 1. Select the most promising node.
        while node.is_fully_expanded() and not node.is_terminal():
            node = select_best_child(node)

        # 2. Expand one unexplored move.
        if not node.is_terminal():
            node = expand(node)

        # 3. Simulate the game from the new node.
        node, reward = random_playout(node, root.player)

        # 4. Update the path with the simulation result.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    return most_visited_child(root).action
        
