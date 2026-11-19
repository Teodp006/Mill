from agent import Agent
import agent
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction
from game import Game
from human_player import HumanPlayer
from match_stats import MatchSession
from match_stats import close_session
from match_stats import print_match_result
import numpy as np
import random

DEBUG = True
# Batch games use BASE_SEED, BASE_SEED+1, ... so the same run can be repeated.
BASE_SEED = 2026
MENU_DIFFICULTIES = list(agent.DIFICULTIES._member_names_)


# Auxiliar function to ask the user to provide a level of difficulty for the AI player
def choose_difficulty(player_number):
    while True:
        difficulty = input(
            f"Choose the difficulty for AI ({player_number}) "
            "(easy, medium, hard or base): ").lower()

        if difficulty not in MENU_DIFFICULTIES:
            print("Unsupported difficulty. Choose easy, medium, hard or base.")
            continue

        return difficulty




def main():
    answer = input(
        "Welcome to the Mill game\n"
        "You can choose one of the following options:\n"
        "a) Play against AI\n"
        "b) See the AI play against another AI\n"
        "c) Simulate matches between AI and AI or a dumb player\n"
    )

    env = None

    # We support some different modes
    match answer:
        case "a": 
            # User playing interactively against an AI (with render mode)
            difficulty = choose_difficulty(2)

            # Create the players using the classes, for the human we use the env even though we will create it later on
            players = (HumanPlayer(1, env), Agent(2, difficulty))

            player_names = ("human", f"AI {difficulty}")
            mode = "watch"
        case "b":
            # Two AI's paying against each other (with render mode)

            # Dificulty selection phase
            difficulty_1 = choose_difficulty(1)
            difficulty_2 = choose_difficulty(2)

            # Create the players with the chosen dificulties
            players = (Agent(1, difficulty_1), Agent(2, difficulty_2))

            #
            player_names = (f"AI {difficulty_1}", f"AI {difficulty_2}")
            mode = "watch"
        case "c": # Making simulations between AI's or one AI and a dumb player that provides random movements
            while True:

                # Match selection
                match_type = input(
                    "Simulate matches between AI and:\n"
                    "1) another AI\n"
                    "2) a dumb player\n").strip()

                match match_type:
                    case "1":
                        # If two IA's are playing  we choose their dificulty and create them
                        difficulty_1 = choose_difficulty(1)
                        difficulty_2 = choose_difficulty(2)
                        players = (Agent(1, difficulty_1), Agent(2, difficulty_2))

                        player_names = (f"AI {difficulty_1}", f"AI {difficulty_2}")
                        break
                    case "2":
                        # If the dumb player is playing it is encoded as None, so when "no player" meaning is a dumb player, the game performs a random move
                        difficulty = choose_difficulty(1)
                        players = (Agent(1, difficulty), None)

                        player_names = (f"AI {difficulty}", "dumb player")
                        break
                    case _:
                        print("Choose 1 or 2.")
            mode = "simulate"
        case _:
            raise ValueError("Unsupported option. Select a, b or c")


    # Starting turn selection:
    start = input(
        f"You want to start the match {player_names[0]} (1) "
        f"or {player_names[1]} (2)? "
    )

    if start not in ["1", "2"]:
        raise ValueError("Choose player 1 or player 2")

    # We reoder the players
    if start == "2":
        players = (players[1], players[0])

        # For the agents we need to update their "turn" as they are changed
        for player in players:
            if isinstance(player, (Agent, HumanPlayer)):
                player.player = agent._opponent_player(player.player)

    
    print(
        f"Okay {player_names[int(start) - 1]} ({start}) starting the game "
        f"against {player_names[2 - int(start)]} ({3 - int(start)})."
    )

    # Initialice the match stats managing system
    session = MatchSession()

    try:
        # This mode is to use the rendering mode, when you want either to play as a human or to see two artificial players face each other
        if mode == "watch":
            keep_playing = True
            while keep_playing:

                # Create the environment with render
                env = UserInteraction(mill.env(render_mode="human"))

                # Pass the environment to the human player so it can interact with the screen
                for player in players:
                    if isinstance(player, HumanPlayer):
                        player.env = env

                # Start the actual game
                record = Game(env, *players).play()
                session.add(record)
                print_match_result(record)

                if record["end_reason"] in ("quit", "truncation"):
                    break

                # While interacting we always ask 
                while True:
                        answer = input("Do you wanna play another match? (y/n): ").strip().lower()
                        if answer in ("y", "yes", "s", "si", "sí"):
                            keep_playing = True
                            break
                        if answer in ("n", "no"):
                            keep_playing = False
                            break
                        print("Choose y or n.")
            
            close_session(session)
        else:
            # First we ask the number of simulations they want to perform
            match_count = 0;

            while True:
                match_count = input("How many matches do you want to simulate? ").strip()

                if match_count.isdigit() and int(match_count) > 0:
                    match_count = int(match_count)
                    break

                print("Choose a positive integer.")

            # We also ask the user if the want the stats to be saved, so while the processing can be in second plane without further interaction
            save_stats = False
            while True:
                answer = input(f"You want to save the stats from the {match_count} match{"es" if match_count > 1 else ""}? (y/n): ").strip().lower()
                if answer in ("y", "yes", "s", "si", "sí"):
                    save_stats = True
                    break
                if answer in ("n", "no"):
                    save_stats = False
                    break
                print("Choose y or n.")

            # Simulations run without a window
            for game_number in range(1, match_count + 1):
                seed = BASE_SEED + game_number - 1
                random.seed(seed)
                np.random.seed(seed)

                # We just print the information of the process
                print(f"[{game_number}/{match_count}] playing (seed {seed})...", flush=True)

                # Create the environment without render
                env = mill.env(render_mode=None)

                # Start the actual game
                record = Game(env, *players).play(seed=seed)

                # To save the stats of the game
                session.add(record)

                # Only print each game result if the researcher needs for any debug or test purpose
                if DEBUG:
                    print_match_result(record)

            # Once the simulatons are finished 
            close_session(session, save=save_stats)

    except KeyboardInterrupt:
        print()
        close_session(session)


if __name__ == "__main__":
    main()
