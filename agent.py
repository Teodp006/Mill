from famnit_gym.envs.mill.mill_model import MillModel
from enum import Enum

import random
import math

# Scores for useful moves. The playout itself only returns 1 for a win and -1 for a loss.
REWARD_CAPTURE = 3
REWARD_MILL = 4
REWARD_BLOCK = 3
REWARD_THREAT_CAPTURE = 2

# Dificulties suported and number of operations for each one where base is just normal Montecarlo.
DIFICULTIES = Enum("Dificulties", ["easy", "medium", "hard", "base"])
ITERATIONS = [20, 100, 1000, 100]


def _opponent_player(current_player):
    return 2 if current_player == 1 else 1

#
def _select_best_child(node):

    max_uct = -float('inf')
    best_child = None

    for child in node.children:
        if _uct_value(child) > max_uct:
            max_uct = _uct_value(child)
            best_child = child

    return best_child

def _uct_value(node, exploration_weight=math.sqrt(2)):
    # In case the node has never been visited we try to explore it
    if node.visits == 0:
        return float('inf')

    # UCT(i) = X̄ + c × √(ln(N)/nᵢ)

    # Exploration: prorices trying new paths (c = √2)
    exploration = exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)

    # Explotation: priorices known paths (X̄)
    exploitation = node.total_reward / node.visits

    return exploration + exploitation


class Node:
    # Store one position in the search tree.
    def __init__(self, model, player, parent=None, action=None, allowed_actions=None):
        self.model = model # State the node is representing
        self.player = player # Player that has the turn in that state (so the next one that is playing)
        self.parent = parent # Parent node
        self.action = action # Action that created this node from the parent node
        self.allowed_actions = allowed_actions # 
        self.children = [] # List of expanded children nodes
        self.visits = 0 # Number of visits to the node
        self.total_reward = 0.0 # Total reward of the node (wins - losses)
        self.expanded_actions = set() # Set off unique actions that have been explored to create new children
        self._legal_moves = None # Moves that can be performed from this node (every move that is legal)

    # Return legal moves that are allowed for this node.
    def legal_moves(self):
        if self._legal_moves is None:
            self._legal_moves = self.model.legal_moves(self.player)
            if self.allowed_actions is not None:
                allowed = {tuple(move) for move in self.allowed_actions}
                self._legal_moves = [move for move in self._legal_moves if tuple(move) in allowed]
        return self._legal_moves

    # Check whether the game has ended.
    def is_terminal(self):
        return self.model.game_over() or not self.legal_moves()

    # Check whether every legal move has been explored.
    def is_fully_expanded(self):
        return len(self.expanded_actions) == len(self.legal_moves())


class Agent:
    def __init__(self, player, dificulty):
        if player not in [1, 2]:
            raise ValueError(f"Unsuported player index: {player}, the agent should play as player 1 or 2")
        if dificulty not in ["easy", "medium", "hard", "base"]:
            raise ValueError(f"Unsuported dificulty mode: {dificulty}, the provided dificulties are [\"easy\", \"medium\", \"hard\", \"base\"]")

        self.player = player
        self.dificulty = dificulty
        self.iterations = ITERATIONS[DIFICULTIES[dificulty].value - 1]

    def play(self, original_model):
        return self._montecarlo(self.iterations, original_model.clone())

    # Count the player's open mills.
    def _count_threats(self, model, player):
        board = model._board
        threats = 0
        for mill in MillModel.mills:
            values = [board[position] for position in mill]
            if values.count(player) == 2 and values.count(0) == 1:
                threats += 1
        return threats

    # Check whether a piece on this position completes a mill.
    def _forms_mill(self, model, player, position):
        board = model._board
        return any(position in mill and all(board[index] == player for index in mill) for mill in MillModel.mills)

    # Count open mills that include a position.
    def _threats_containing(self, model, player, position):
        board = model._board
        return sum(position in mill and [board[index] for index in mill].count(player) == 2 and [board[index] for index in mill].count(0) == 1 for mill in MillModel.mills)

    # Score a move using immediate tactical benefits.
    def _tactical_score(self, model, player, move):
        src, dst, take = move
        opponent = _opponent_player(player)
        board = model._board

        # Count opponent threats before applying the move.
        opponent_threats_before = self._count_threats(model, opponent)
        captured_threats = self._threats_containing(model, opponent, take) if take else 0

        # Save the affected positions before simulating the move.
        original_src = board[src]
        original_dst = board[dst]
        original_take = board[take] if take else 0

        # Apply the move temporarily. Placing moves use src == 0, so dst must always be set.
        if src:
            board[src] = 0
        board[dst] = player
        if take:
            board[take] = 0

        score = 0

        # Reward moves that create a mill.
        if self._forms_mill(model, player, dst):
            score += REWARD_MILL

        # Reward moves that remove opponent threats.
        blocked_threats = opponent_threats_before - self._count_threats(model, opponent)
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
    def _prioritized_choice(self, model, player, moves):
        scores = [self._tactical_score(model, player, move) for move in moves]
        weights = [1 + max(score, 0) for score in scores]
        return random.choices(moves, weights=weights, k=1)[0]

    # Base explores uniformly. Any other dificulty weights the tactical moves.
    def _choose_move(self, model, player, moves):
        if self.dificulty == "base":
            return random.choice(moves)
        return self._prioritized_choice(model, player, moves)

    # Check whether a move completes a mill.
    def _move_forms_mill(self, model, player, move):
        candidate = model.clone()
        candidate.make_move(player, move)
        return self._forms_mill(candidate, player, move[1])

    # Check whether the player can complete a mill.
    def _has_immediate_mill(self, model, player):
        return any(self._move_forms_mill(model, player, move) for move in model.legal_moves(player))

    # Keep winning moves and block immediate opponent mills.
    def _tactical_root_moves(self, model, player, moves):
        mill_moves = [move for move in moves if self._move_forms_mill(model, player, move)]
        if mill_moves:
            return mill_moves

        opponent = _opponent_player(player)
        if not self._has_immediate_mill(model, opponent):
            return moves

        blocking_moves = []
        for move in moves:
            candidate = model.clone()
            candidate.make_move(player, move)
            if not self._has_immediate_mill(candidate, opponent):
                blocking_moves.append(move)

        return blocking_moves or moves

    def _expand(self, node):
        # The possible moves represent the transactions from the parent to its children so two children don't represent the same move
        unexpanded_moves = [move for move in node.legal_moves() if tuple(move) not in node.expanded_actions]

        # We select the expected next move following the policy of this dificulty
        move = self._choose_move(node.model, node.player, unexpanded_moves)
        node.expanded_actions.add(tuple(move))

        # We create a new model for the child node
        new_model = node.model.clone()

        # We perform the move in the new model
        new_model.make_move(node.player, move)

        # We create the new node
        next_node = Node(model=new_model, player=_opponent_player(node.player), parent=node, action=move)

        # We add the new node to the parents child list
        node.children.append(next_node)

        # Return the choosen child
        return next_node

    def _random_playout(self, node):
        # The actual node copies does a simulation from his current state
        actualmodel = node.model.clone()

        # We start from the player who has the turn
        current_player = node.player

        # We simulate the match until it finishes
        while not actualmodel.game_over():
            moves = actualmodel.legal_moves(current_player)
            if not moves:
                break

            # Base uses a random move. The other dificulties use the tactical weights.
            move = self._choose_move(actualmodel, current_player, moves)
            actualmodel.make_move(current_player, move)

            # We switch the turn of the players
            current_player = _opponent_player(current_player)

        # The player who would move next lost when the previous move ended the game.
        # Win end state gives 1. Lose end state gives -1.
        winner = _opponent_player(current_player)
        reward = 1 if winner == self.player else -1
        return node, reward

    def _most_visited_child(self, node):
        max_visited = -float('inf')
        best_child = None
        for child in node.children:
            if child.visits > max_visited:
                max_visited = child.visits
                best_child = child
        return best_child

    def _montecarlo(self, iterations, model):
        # Initialice the root node
        root_moves = model.legal_moves(self.player)
        root_moves = self._tactical_root_moves(model, self.player, root_moves)
        root = Node(model=model, player=self.player, allowed_actions=root_moves)

        for _ in range(iterations):
            node = root

            # 1.Selection (Exploration VS Explotation)
            while node.is_fully_expanded() and not node.is_terminal():
                node = _select_best_child(node)  # The highest UCT value.

            # 2.Expansion
            if not node.is_terminal():
                node = self._expand(node)  # Add one move that this node has not tried yet.

            # 3.Simulation
            node, reward = self._random_playout(node)  # From the root player's point of view.

            # 4.Backpropagation
            while node is not None:
                node.visits += 1
                node.total_reward += reward
                node = node.parent

        return self._most_visited_child(root).action
