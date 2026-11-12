from time import sleep

from famnit_gym.envs import mill
from HumanPlayer import HumanPlayer
from agent import Agent


class Game:
    def __init__(self, environment, player_1=None, player_2=None):
        self.environment = environment
        self.players = (player_1, player_2)

    def play(self):
        # Create and reset the environment.
        self.environment.reset()

        for current_agent in self.environment.agent_iter():
            # Get the state of the current player.
            _, _, termination, truncation, _ = (
                self.environment.last()
            )

            # The game ends when one player loses.
            if termination:
                print(f"{current_agent} lost the game!")
                break

            # The game is truncated if it is too long or the window is closed.
            if truncation:
                print("The game is too long!")
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

                # During the interaction the user terminated the game we finish
                if player.quit_requested:
                    break
            
            # The IA chooses a move using the current model to search for the best move
            elif isinstance(player, Agent):
                model = mill.transition_model(self.environment.unwrapped)
                move = player.play(model)
                sleep(2)

            # The user quit during the interaction.
            

            # Make the chosen move.
            self.environment.step(move)

        # Close the environment.
        self.environment.close()
