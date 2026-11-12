import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.envs.mill.mill_model import MillModel
from enum import Enum

import random
import math

REWARD_WIN = 10
REWARD_CAPTURE = 4
PUNISHMENT_CAPTURE = 3
PUNISHMENT_DRAW = 2
OPP_MISTAKE_CHANCE = 0.2
MAX_ROLLOUT_MOVES = 200
DIFICULTIES = Enum("Dificulties", ["easy", "medium", "hard"])
MISTAKE_CHANCE = [0.7, 0.4, 0]
ITERATIONS = [20, 100, 1000]


class Agent:
    def __init__(self, player, dificulty):
        if player not in [1, 2]:
            raise ValueError(f"Unsuported player index: {player}, the agent should play as player 1 or 2")
        self.player = player
        if dificulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Unsuported dificulty mode: {dificulty}, the provided dificulties are [\"easy\", \"medium\", \"hard\"]")
        self.mistake_chance = MISTAKE_CHANCE[DIFICULTIES[dificulty].value - 1]
        self.iterations = ITERATIONS[DIFICULTIES[dificulty].value - 1]

    def play(self, original_model):
        model = original_model.clone()

        # Try the trivial winning move: if the other player has 3 pieces and we can do a capture just do it without running Montecarlo
        if model.get_phase(_opponent_player(self.player)) == "flying" and random.random() > self.mistake_chance:
            for move in model.legal_moves(self.player):
                if move[2] > 0:
                    return move
                
        # Try the trivial saving move: if the other player can do a capture and we have 3 pieces try to block that
        if model.get_phase(self.player) == "flying" and random.random() > self.mistake_chance:
            for [src, dst, capture] in model.legal_moves(_opponent_player(self.player)):
                if capture > 0:
                    legal_moves = model.legal_moves(self.player)
                    for move in legal_moves: # We iterate to see if one of them leads to a not lost game
                        new_model = model.clone()
                        new_model.make_move(move, self.player)
                        if move[1] == dst and new_model.get_phase(self.player) != "lost":  
                            return move
        # In case there is no saving move the most probable thing is that the opponent actually wins us, so we could do a random move here but the enemy is not always optimal so there might be an scenerio were we have an extra chance to win
        
        # If none off the trivial moves can be performed, then simply use montecarlo predictions to find the best possible move
        if random.random() > self.mistake_chance:
            return self._montecarlo(self.iterations, model)
        else:
            return None

    def _expand(self, node, model):

        # The possible moves represent the transactions from the parent to it's children for every new "fresh" child so we make sure two child don't represent the same exact space
        posible_moves = [move for move in node.legal_moves if not any(child.action == move for child in node.children)]

        # We select the expected next move following the policy of the agent and the suposed opponent
        move = self._roll_out_move(posible_moves, node.model, node.player, mistake_chance = -1) # For building the tree we supose every player plays optimally

        # We create a new model for the child node
        new_model = node.model.clone()

        # We perform the move in the new model
        new_model.make_move(node.player, move)

        # We create the new model
        next_node = Node(model=new_model, player=_opponent_player(node.player), parent=node, action=move)

        # We add the new node to the parents child list
        node.children.append(next_node)

        # Return the choosen child
        return next_node

    # Policy to choose next move favorably when there is a possible capture (the agent supose the opponent has a chance to do a non optimal choose, so we sometimes predicts a random move can be performed by the opponent)
    def _roll_out_move(self, moves, model, current_player, mistake_chance = 0.2):

        capturing_moves = []
        opponent_mills = []

        # In case we are the opponent player turn we supose it may do mistake_chance% bad moves
        good_move = True if random.random() > OPP_MISTAKE_CHANCE else False

        # We look if there is any possible move from this stage where we can do a capture
        if current_player == self.player or good_move:
            capturing_moves = [move for move in moves if move[2] > 0]

        # It is important to have in mind the possible mills the opponent can create in his turn
        opponent_mills = self._opponent_future_mills(model)

        if capturing_moves:
            if opponent_mills: # We choose to capture any possible piece that can form a mill in the opponents turn
                # We define the strategic moves such as those that allow our agent to capture an oponent piece that could make an mill
                strategic_moves = [move for mill in opponent_mills for move in capturing_moves if move[2] in mill]
                if strategic_moves:
                    # We order the opponent pieces that can form a mill by the amount of mills they can form
                    best_positions = self._most_important_position(opponent_mills)
                    for index in range(len(best_positions) - 1):
                        # We define the list of best positions we can capture with any of our capturing moves
                        capturable_positions = [position for position in best_positions[index] if any(position == move[2] for move in capturing_moves)]
                        if capturable_positions:
                            strategic_moves = [move for move in strategic_moves if move[2] in capturable_positions]
                            break
                    return random.choice(strategic_moves)
            return random.choice(capturing_moves)
        else:
            if opponent_mills or good_move: # We choose to move our piece to any possible future mill of the opponent
                blocking_moves = [move for mill in opponent_mills for move in moves if move[1] in mill]
                if blocking_moves:
                    return random.choice(blocking_moves)
            return random.choice(moves)

    # Method that returns the lists of positions where the opponent can do a mill in the current move so the agent can perform a block
    def _opponent_future_mills(self, model):
        board = [0] + model.get_state()
        opponent_player = _opponent_player(self.player)
        opponent_mills = []
        for mill in MillModel.mills:
            # We check two enemy pieces in a row with one free space
            if len([board[position] for position in mill if board[position] == opponent_player]) == 2 and any(board[position] == 0 for position in mill):
                opponent_mills.append(mill)
        return opponent_mills

    # Used to count how many times a piece shows up at the opponent mills, used also to calculate the most important piece to capture for the oponent side
    def _most_important_position(self, mills):
        positions = [position for mill in mills for position in mill]
        if not positions:
            return []

        # Count how many a piece or hole forms a future mill
        repetitions = {position: positions.count(position) for position in positions}

        # Return groups of pieces/holes ordered from the most important to the least important
        repetition_values = sorted(set(repetitions.values()), reverse=True)
        return [
            [
                position
                for position in repetitions
                if repetitions[position] == repetition
            ]
            for repetition in repetition_values
        ]
            



    def _random_playout(self, node):
        # The actual node copies does a simulation from his current state
        actualmodel = node.model.clone()

        # We start from the player who has the turn
        current_player = node.player

        # The reward asociated to the match (including wins and good moves (performed as captures))
        reward = 0

        # We simulate the match until it does not finishes
        for _ in range(MAX_ROLLOUT_MOVES):

            # If the match has ended we exit the loop
            if actualmodel.game_over():
                break

            # We choose a next good move
            move = self._roll_out_move(actualmodel.legal_moves(current_player), actualmodel, current_player)

            # From the perspective of the root player (the agent) we check if the current player performed a capture
            if(move[2] > 0):
                # If out opponent performed a capture we substract some reward because is not a good move for us
                if(current_player == self.player):
                    reward += REWARD_CAPTURE
                # Similarely if we perform it is a good move
                else:
                    reward -= PUNISHMENT_CAPTURE

            # We perform that move in our new model
            actualmodel.make_move(current_player, move)

            # We switch the turn of the players
            current_player = _opponent_player(current_player)


        # We check the end state of the match from the perspective of the root
        agent_state = actualmodel.get_phase(self.player)
        opponent_state = actualmodel.get_phase(_opponent_player(self.player))

        if(agent_state == "lost") and (opponent_state != "lost"):
            reward -= REWARD_WIN # Lose end state ["lost", "moving/placing/flying", False]
        elif (agent_state != "lost") and (opponent_state == "lost"):
            reward += REWARD_WIN # Win end state ["moving/placing/flying", "lost", False]
        else:
            reward -= PUNISHMENT_DRAW
    
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
        root = Node(model=model, player=self.player)

        for _ in range(iterations):
            node = root

            # 1.Selection (Exploration VS Explotation)
            while node.isFullyExpanded() and not node.is_terminal:
                node = _select_best_child(node)  # The highest UCT value.

            # 2.Expansion
            if not node.is_terminal:
                node = self._expand(node, model)  # Add a randomly chosen child.

            # 3.Simulation
            node, reward = self._random_playout(node)  # From the root player's point of view.

            # 4.Backpropagation
            while node.parent is not None:
                node.visits += 1
                if node.parent.player == root.player:  # The player that made the move is the agent (because the father of the node is the one that did the move)
                    node.total_reward += reward
                else:  # The player that made the move is not the agent
                    node.total_reward -= reward
                node = node.parent

            # We update also the visits to the root node
            root.visits += 1

        for child in root.children:
            print(
                "Move:", child.action,
                "Visits:", child.visits,
                "Reward:", child.total_reward,
                "Average:", child.total_reward / child.visits
                if child.visits > 0 else 0
            )

        best_child = self._most_visited_child(root)
        print("Selected move:", best_child.action)

        return self._most_visited_child(root).action


def _opponent_player(current_player):
        return 2 if current_player == 1 else 1


def _select_best_child(node):

    max_uct = -float('inf')
    best_child = None

    for child in node.children:
        if _uct_value(child) > max_uct:
            max_uct = _uct_value(child)
            best_child = child

    return best_child

def _uct_value(node, exploration_weight=math.sqrt(2)):
    # In case the node has never been visited
    if node.visits == 0:
        return float("inf")

    exploration = exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
    exploitation = node.total_reward / node.visits

    return exploration + exploitation


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
        self.is_terminal = (self.model.game_over() or not self.legal_moves)

    def isFullyExpanded(self):
        return len(self.children) == len(self.legal_moves)
