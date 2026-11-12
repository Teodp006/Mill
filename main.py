from agent import Agent
import agent
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction
from game import Game
from HumanPlayer import HumanPlayer

# Auxiliar function to ask the user to provide a level of dificulty for the IA player
def ask_difficulty(player_number):
    while True:
        difficulty = input(
            f"Choose the difficulty for IA ({player_number}) "
            "(easy, medium or hard): ").lower()
        
        # Check if the dificulty is available
        if difficulty in agent.DIFICULTIES.__members__:
            return difficulty
        print("Unsupported difficulty. Choose easy, medium or hard.")


def main():
    answer = input(
        "Welcome to the Mill game\n"
        "You can choose one of the following options:\n"
        "a) Play against IA\n"
        "b) Make one IA play against another IA\n"
        "c) Make the IA play against a dumb player\n"
    )

    env = None


    match answer:
        case "a":
            difficulty = ask_difficulty(2)
            players = (
                HumanPlayer(1, env),
                Agent(2, difficulty),
            )
            player_names = ("human", f"IA {difficulty}")
        case "b":
            difficulty_1 = ask_difficulty(1)
            difficulty_2 = ask_difficulty(2)
            players = (
                Agent(1, difficulty_1),
                Agent(2, difficulty_2),
            )
            player_names = (f"IA {difficulty_1}", f"IA {difficulty_2}")
        case "c":
            difficulty = ask_difficulty(1)
            players = (
                Agent(1, difficulty),
                None,
            )
            player_names = (f"IA {difficulty}", "dumb player")
        case _:
            env.close()
            raise ValueError("Unsupported option. Select a, b or c")

    start = input(
        f"You want to start the match {player_names[0]} (1) "
        f"or {player_names[1]} (2)? "
    )
    if start not in ["1", "2"]:
        raise ValueError("Choose player 1 or player 2")

    if start == "2":
        players = (players[1], players[0])

        # If we reorder the players turns we have to make sure that we change the turn of the IA
        for player in players:
            if isinstance(player, Agent) or isinstance(player, HumanPlayer):
                player.player = agent._opponent_player(player.player)

    print(
        f"Okay {player_names[int(start) - 1]} ({start}) starting the game "
        f"against {player_names[2 - int(start)]} ({3 - int(start)})."
    )

    # We create the environment now to show up in the screen and update any Human the new environment so they can interact with it
    env = UserInteraction(mill.env(render_mode="human"))
    for player in players:
        if isinstance(player, HumanPlayer):
            player.env = env


    Game(env, *players).play()


if __name__ == "__main__":
    main()
