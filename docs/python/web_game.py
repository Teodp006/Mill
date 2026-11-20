"""Bridge between the page and the lab code.

The browser never reimplements the rules or the search. It calls handle(),
and this module talks to MillModel and Agent, the same classes used by main.py.
"""

import json

from agent import Agent
from famnit_gym.envs.mill.mill_model import MillModel

model = None
agent = None
human = 1
ai_player = 2
turn = 1


def _public_state(last_move=None):
    over = bool(model.game_over())
    active = None if over else turn
    legal = []
    if active == human:
        legal = [list(move) for move in model.legal_moves(human)]

    winner = None
    if model.get_phase(1) == "lost":
        winner = 2
    elif model.get_phase(2) == "lost":
        winner = 1

    return {
        "board": list(model.get_state()),
        "phases": [model.get_phase(1), model.get_phase(2)],
        "holding": [
            model._player[1]["pieces_holding"],
            model._player[2]["pieces_holding"],
        ],
        "playing": [
            model._player[1]["pieces_playing"],
            model._player[2]["pieces_playing"],
        ],
        "turn": active,
        "human": human,
        "ai": ai_player,
        "over": over,
        "winner": winner,
        "legal": legal,
        "move": None if last_move is None else list(last_move),
        "mills": MillModel.mills,
    }


def new_game(human_player, difficulty):
    global model, agent, human, ai_player, turn

    human = int(human_player)
    if human not in (1, 2):
        raise ValueError("El humano tiene que ser el jugador 1 o el 2.")

    ai_player = 2 if human == 1 else 1
    model = MillModel()
    # Agent.__init__ checks the difficulty names used in the lab.
    agent = Agent(ai_player, str(difficulty))
    turn = 1
    return json.dumps(_public_state())


def play_human(move):
    global turn

    if model is None:
        raise RuntimeError("No hay partida.")
    if model.game_over() or turn != human:
        raise RuntimeError("Ahora no te toca.")

    chosen = [int(part) for part in move]
    legal = {tuple(item) for item in model.legal_moves(human)}
    if tuple(chosen) not in legal:
        raise ValueError(f"Movimiento ilegal: {chosen}")

    model.make_move(human, chosen)
    turn = None if model.game_over() else ai_player
    return json.dumps(_public_state(chosen))


def play_ai():
    global turn

    if model is None:
        raise RuntimeError("No hay partida.")
    if model.game_over() or turn != ai_player:
        return json.dumps(_public_state())

    # This is the call that runs Monte Carlo inside agent.py.
    chosen = [int(part) for part in agent.play(model)]
    model.make_move(ai_player, chosen)
    turn = None if model.game_over() else human
    return json.dumps(_public_state(chosen))


def handle(name, payload):
    name = str(name)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if name == "new_game":
        return new_game(payload["human"], payload["difficulty"])
    if name == "play_human":
        return play_human(payload["move"])
    if name == "play_ai":
        return play_ai()
    raise ValueError(f"Operación desconocida: {name}")
