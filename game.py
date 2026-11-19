from time import perf_counter

from famnit_gym.envs import mill
from human_player import HumanPlayer
from agent import Agent
from match_stats import GameRecorder
from match_stats import pieces_remaining


class Game:
    def __init__(self, environment, player_1=None, player_2=None):
        self.environment = environment
        self.players = (player_1, player_2)
        

    def play(self, seed=None):
        # The recorder follows this match until it ends or the user leaves.
        recorder = GameRecorder(self.players[0], self.players[1])
        end_reason = None
        winner_player = None

        # Create and reset the environment. Batch mode can pass a seed for replayable runs.
        if seed is None:
            self.environment.reset()
        else:
            self.environment.reset(seed=seed)
        for player in self.players:
            if isinstance(player, HumanPlayer):
                player.quit_requested = False

        try:
            for current_agent in self.environment.agent_iter():
                # Get the state of the current player.
                _, _, termination, truncation, _ = (
                    self.environment.last()
                )

                # The game ends when one player loses.
                if termination:
                    print(f"{current_agent} lost the game!")
                    loser = 1 if current_agent == "player_1" else 2
                    winner_player = 3 - loser
                    model = mill.transition_model(self.environment.unwrapped)
                    
                    # We save the lossing reason for our stats
                    if pieces_remaining(model, loser) < 3:
                        end_reason = "pieces"
                    else:
                        end_reason = "immobilization"
                    break

                # The game is truncated if it is too long or the window is closed.
                if truncation:
                    print("The game is too long!")
                    end_reason = "truncation"
                    break

                # The agent name tells us which player has the current turn.
                player_number = 1 if current_agent == "player_1" else 2
                player = self.players[player_number - 1]

                # If is the turn of the dumb player, no implementation is provided so perform a random move
                if player is None:
                    self.environment.step(None)
                    continue

                # The human interacts with the board directly.
                if isinstance(player, HumanPlayer):
                    move = player.play()

                    # Escape and the window close both stop the match here.
                    if player.quit_requested:
                        print("User quit interactively!")
                        end_reason = "quit"
                        break

                # The AI chooses a move using the current model to search for the best move
                elif isinstance(player, Agent):
                    model = mill.transition_model(self.environment.unwrapped)

                    # We calculate the computational time of the agent
                    started = perf_counter()
                    move = player.play(model)

                    # The recorder adds the information to the match stats
                    recorder.add_decision(player_number, perf_counter() - started)

                # Make the chosen move.
                self.environment.step(move)
        # We allways save the stats at the end of the program 
        finally:
            # Read the board before closing. Quit and truncation still have a margin.
            model = mill.transition_model(self.environment.unwrapped)
            if end_reason is None:
                end_reason = "quit"

            # The recorder stores all the information before closing the app
            record = recorder.finish(model, end_reason, winner_player)

            self.environment.close()

        return record
