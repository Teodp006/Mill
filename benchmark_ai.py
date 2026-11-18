"""
Outputs (saved after each game):
  *_games.csv   -- one row per attempted game, including errors and timeouts.
  *_summary.csv -- one row per contender; rates use completed games and draws.

Metric definitions:
  captures: opponent pieces actually removed from the board.
  mill_blockades: moves occupying an opponent's legal immediate mill target.
  mill_threats_blocked: distinct mill/target opportunities blocked by those moves.
  mills_formed: newly completed mill lines (a double mill counts as two).
  immobilization_wins: terminal wins while the loser still has >= 3 pieces.
  material_margin_pct: (own remaining - opponent remaining) / 9 * 100.
    Remaining includes unplaced pieces. This margin can be negative in a win
    by immobilization; it is a material measure, not a measure of game quality.
  win_margin_pp: win percentage minus loss percentage, in percentage points.
"""

import contextlib
import csv
from datetime import datetime
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
from time import perf_counter


# Experiment settings


DIFFICULTY_A = "easy"
DIFFICULTY_B = "medium"
NUM_GAMES = 40
BASE_SEED = 2026

# A starts as player 1 in odd games, B in even games. Player 1 starts.
MAX_HALF_MOVES = 400  # Individual actions, not pairs of turns.
GAME_TIMEOUT_SECONDS = 600  # A stuck search is recorded as a timeout.
INITIAL_PIECES_PER_PLAYER = 9
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_results"


def load_game():
    """Use the same installed game and Agent class as main.py."""
    import numpy as np
    from agent import Agent
    from famnit_gym.envs import mill
    from famnit_gym.envs.mill.mill_model import MillModel

    return np, Agent, mill, MillModel


def new_result(game_number):
    a_player = 1 if game_number % 2 else 2
    result = {
        "game": game_number,
        "seed": BASE_SEED + game_number - 1,
        "difficulty_a": DIFFICULTY_A,
        "difficulty_b": DIFFICULTY_B,
        "a_player": a_player,
        "b_player": 3 - a_player,
        "starting_contender": "a" if a_player == 1 else "b",
        "status": "error",
        "winner": "",
        "winner_difficulty": "",
        "end_reason": "",
        "half_moves": 0,
        "duration_seconds": 0.0,
        "error": "",
    }
    for side in ("a", "b"):
        for metric in ("moves", "captures", "mill_blockades",
                       "mill_threats_blocked", "mills_formed",
                       "immobilization_wins", "random_fallbacks"):
            result[f"{side}_{metric}"] = 0
        # Leave board counts unknown until the environment was inspected.
        result[f"{side}_pieces_on_board"] = ""
        result[f"{side}_pieces_remaining"] = INITIAL_PIECES_PER_PLAYER
        result[f"{side}_material_margin_pct"] = 0.0
        result[f"{side}_decision_seconds"] = 0.0
        result[f"{side}_mean_decision_ms"] = ""
    return result


def update_totals(result, board=None):
    for side, opponent in (("a", "b"), ("b", "a")):
        result[f"{side}_pieces_remaining"] = (
            INITIAL_PIECES_PER_PLAYER - result[f"{opponent}_captures"]
        )
        if board is not None:
            result[f"{side}_pieces_on_board"] = board.count(result[f"{side}_player"])
        moves = result[f"{side}_moves"]
        if moves:
            result[f"{side}_mean_decision_ms"] = round(
                result[f"{side}_decision_seconds"] * 1000 / moves, 3
            )
    margin = (
        result["a_pieces_remaining"] - result["b_pieces_remaining"]
    ) / INITIAL_PIECES_PER_PLAYER * 100
    result["a_material_margin_pct"] = round(margin, 3)
    result["b_material_margin_pct"] = round(-margin, 3)


def formed_mills(board, player, mills):
    return {
        tuple(line) for line in mills
        if all(board[position] == player for position in line)
    }


def immediate_mill_targets(model, player, mills):
    """Deduplicate capture variants of the same legal mill completion."""
    targets = set()
    board = [0] + list(model.get_state())
    for src, dst, take in model.legal_moves(player):
        if not take:
            continue
        candidate = list(board)
        if src:
            candidate[src] = 0
        candidate[dst] = player
        for line in mills:
            if dst in line and all(candidate[position] == player for position in line):
                targets.add((tuple(line), int(dst)))
    return targets


def play_game(game_number, publish=None):
    """Run a headless game; return measurements even if an agent raises."""
    result = new_result(game_number)
    started = perf_counter()
    environment = None
    acting_side = ""
    quiet = None
    try:
        np, Agent, mill, MillModel = load_game()
        random.seed(result["seed"])
        np.random.seed(result["seed"])
        environment = mill.env(render_mode=None)
        environment.reset(seed=result["seed"])
        sides = {result["a_player"]: "a", result["b_player"]: "b"}
        agents = {
            result["a_player"]: Agent(result["a_player"], DIFFICULTY_A),
            result["b_player"]: Agent(result["b_player"], DIFFICULTY_B),
        }
        quiet = open(os.devnull, "w")

        while True:
            model = mill.transition_model(environment.unwrapped)
            board = [0] + list(model.get_state())
            update_totals(result, board)

            # Capture rewards are not terminal rewards. Read both model phases.
            if model.game_over():
                losers = [p for p in (1, 2) if model.get_phase(p) == "lost"]
                if len(losers) != 1:
                    raise RuntimeError("Terminal model does not have exactly one losing player")
                losing_side = sides[losers[0]]
                winning_side = sides[3 - losers[0]]
                result["status"] = "completed"
                result["winner"] = winning_side
                result["winner_difficulty"] = result[f"difficulty_{winning_side}"]
                if result[f"{losing_side}_pieces_remaining"] < 3:
                    result["end_reason"] = "pieces"
                else:
                    result["end_reason"] = "immobilization"
                    result[f"{winning_side}_immobilization_wins"] = 1
                break

            _, _, termination, truncation, _ = environment.last()
            if termination:
                raise RuntimeError("Environment terminated without a terminal model")
            if truncation or result["half_moves"] >= MAX_HALF_MOVES:
                result["status"] = "draw"
                result["winner"] = "draw"
                result["end_reason"] = "environment_truncation" if truncation else "move_limit"
                break

            player = {"player_1": 1, "player_2": 2}[environment.agent_selection]
            opponent = 3 - player
            acting_side = sides[player]
            legal_moves = [list(move) for move in model.legal_moves(player)]
            if not legal_moves:
                raise RuntimeError("Nonterminal player has no legal moves")
            threats = immediate_mill_targets(model, opponent, MillModel.mills)
            mills_before = formed_mills(board, player, MillModel.mills)

            decision_started = perf_counter()
            try:
                # Suppress MCTS diagnostics without storing them in memory.
                with contextlib.redirect_stdout(quiet):
                    move = agents[player].play(model.clone())
            finally:
                result[f"{acting_side}_decision_seconds"] += perf_counter() - decision_started

            fallback = move is None
            if fallback:
                # Match MillEnv.step(None): a uniform NumPy-random legal action.
                move = legal_moves[int(np.random.choice(len(legal_moves)))]
            move = list(move)
            if len(move) != 3 or move not in legal_moves:
                # Do not silently let the environment replace an illegal action.
                raise ValueError(f"Agent returned an illegal move: {move!r}")
            move = [int(position) for position in move]
            environment.step(move)

            after = mill.transition_model(environment.unwrapped)
            after_board = [0] + list(after.get_state())
            captured = board.count(opponent) - after_board.count(opponent)
            if captured != int(move[2] > 0):
                raise RuntimeError("Observed captures do not match the accepted action")
            blocked = sum(target == move[1] for _, target in threats)
            result["half_moves"] += 1
            result[f"{acting_side}_moves"] += 1
            result[f"{acting_side}_captures"] += captured
            result[f"{acting_side}_random_fallbacks"] += int(fallback)
            result[f"{acting_side}_mill_blockades"] += int(blocked > 0)
            result[f"{acting_side}_mill_threats_blocked"] += blocked
            result[f"{acting_side}_mills_formed"] += len(
                formed_mills(after_board, player, MillModel.mills) - mills_before
            )
            update_totals(result, after_board)
            result["duration_seconds"] = round(perf_counter() - started, 3)
            if publish is not None:
                publish(dict(result))

    except Exception as error:
        result["status"] = "error"
        result["end_reason"] = "exception"
        context = f"contender {acting_side.upper()}: " if acting_side else ""
        result["error"] = f"{context}{type(error).__name__}: {error}"
    finally:
        if quiet is not None:
            quiet.close()
        if environment is not None:
            environment.close()
        update_totals(result)
        result["duration_seconds"] = round(perf_counter() - started, 3)
    return result


def game_worker(game_number, connection):
    try:
        result = play_game(
            game_number,
            publish=lambda row: connection.send(("progress", row)),
        )
        connection.send(("finished", result))
    finally:
        connection.close()


def run_game(game_number):
    """Isolate each game so a stuck rollout cannot hang the whole experiment."""
    context = mp.get_context("spawn")  # Also works on Windows.
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=game_worker, args=(game_number, sender))
    result = new_result(game_number)
    started = perf_counter()
    finished = False
    process.start()
    sender.close()
    try:
        while not finished:
            if receiver.poll(0.2):
                try:
                    message, result = receiver.recv()
                except EOFError:
                    break
                finished = message == "finished"
            elif not process.is_alive():
                break
            if not finished and perf_counter() - started >= GAME_TIMEOUT_SECONDS:
                result["status"] = "timeout"
                result["winner"] = ""
                result["winner_difficulty"] = ""
                result["end_reason"] = "game_timeout"
                result["duration_seconds"] = round(perf_counter() - started, 3)
                result["error"] = f"Exceeded {GAME_TIMEOUT_SECONDS} seconds; partial metrics only"
                break
        if not finished and result["status"] != "timeout":
            result["status"] = "error"
            result["end_reason"] = "worker_exit"
            result["error"] = f"Game process exited without a result (exit code {process.exitcode})"
    finally:
        if process.is_alive() and not finished:
            process.terminate()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()
        process.close()
    return result


def percentage(value, total):
    return round(100 * value / total, 3) if total else ""


def mean(values):
    return round(sum(values) / len(values), 3) if values else ""


def wilson_interval(wins, total):
    if not total:
        return "", ""
    z = 1.959963984540054
    proportion = wins / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return round(100 * max(0, center - radius), 3), round(100 * min(1, center + radius), 3)


def summarize(results):
    valid = [row for row in results if row["status"] in ("completed", "draw")]
    summary = []
    for side, opponent in (("a", "b"), ("b", "a")):
        wins = sum(row["winner"] == side for row in valid)
        losses = sum(row["winner"] == opponent for row in valid)
        draws = sum(row["status"] == "draw" for row in valid)
        low, high = wilson_interval(wins, len(valid))
        row = {
            "contender": side,
            "difficulty": DIFFICULTY_A if side == "a" else DIFFICULTY_B,
            "games_planned": NUM_GAMES,
            "games_recorded": len(results),
            "valid_games": len(valid),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "errors": sum(game["status"] == "error" for game in results),
            "timeouts": sum(game["status"] == "timeout" for game in results),
            "win_pct": percentage(wins, len(valid)),
            "loss_pct": percentage(losses, len(valid)),
            "draw_pct": percentage(draws, len(valid)),
            "score_pct": percentage(wins + 0.5 * draws, len(valid)),
            "win_margin_pp": percentage(wins - losses, len(valid)),
            "win_pct_ci95_low": low,
            "win_pct_ci95_high": high,
        }
        for player in (1, 2):
            seated = [game for game in valid if game[f"{side}_player"] == player]
            row[f"valid_games_as_player_{player}"] = len(seated)
            row[f"win_pct_as_player_{player}"] = percentage(
                sum(game["winner"] == side for game in seated), len(seated)
            )
        for metric in ("captures", "mill_blockades", "mill_threats_blocked",
                       "mills_formed", "immobilization_wins", "random_fallbacks"):
            values = [game[f"{side}_{metric}"] for game in valid]
            row[f"total_{metric}"] = sum(values)
            row[f"mean_{metric}"] = mean(values)
        for metric in ("pieces_remaining", "material_margin_pct"):
            row[f"mean_{metric}"] = mean([game[f"{side}_{metric}"] for game in valid])
        row["mean_material_margin_pct_in_wins"] = mean([
            game[f"{side}_material_margin_pct"] for game in valid if game["winner"] == side
        ])
        moves = sum(game[f"{side}_moves"] for game in valid)
        seconds = sum(game[f"{side}_decision_seconds"] for game in valid)
        row["mean_decision_ms"] = round(1000 * seconds / moves, 3) if moves else ""
        row["mean_game_half_moves"] = mean([game["half_moves"] for game in valid])
        row["mean_game_seconds"] = mean([game["duration_seconds"] for game in valid])
        row["base_seed"] = BASE_SEED
        row["max_half_moves"] = MAX_HALF_MOVES
        row["game_timeout_seconds"] = GAME_TIMEOUT_SECONDS
        row["python_version"] = sys.version.split()[0]
        for name in ("agent.py", "hard_agent.py"):
            path = Path(__file__).resolve().parent / name
            row[name.replace(".py", "_sha256")] = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
            )
        summary.append(row)
    return summary


def write_summary(path, results):
    rows = summarize(results)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if NUM_GAMES <= 0 or NUM_GAMES % 2:
        raise SystemExit("Use a positive even NUM_GAMES so both contenders start equally often")
    if MAX_HALF_MOVES <= 0 or GAME_TIMEOUT_SECONDS <= 0:
        raise SystemExit("MAX_HALF_MOVES and GAME_TIMEOUT_SECONDS must be positive")
    if not 0 <= BASE_SEED <= 2**32 - NUM_GAMES:
        raise SystemExit("Seeds must fit NumPy's unsigned 32-bit seed range")
    try:
        _, Agent, _, _ = load_game()
        Agent(1, DIFFICULTY_A)
        Agent(2, DIFFICULTY_B)
    except Exception as error:
        raise SystemExit(
            f"Cannot load the project: {type(error).__name__}: {error}\n"
            "Put this file beside agent.py and hard_agent.py, and use the same "
            "Python environment that runs main.py."
        ) from error

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{run_name}_{DIFFICULTY_A}_vs_{DIFFICULTY_B}"
    games_path = OUTPUT_DIR / f"{prefix}_games.csv"
    summary_path = OUTPUT_DIR / f"{prefix}_summary.csv"
    results = []
    print(f"Running {NUM_GAMES} games: A={DIFFICULTY_A}, B={DIFFICULTY_B}")
    print("Starting sides alternate. No window or artificial turn delay.")
    print(f"Each game can take up to {GAME_TIMEOUT_SECONDS} seconds.", flush=True)

    with games_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(new_result(1)))
        writer.writeheader()
        output.flush()
        try:
            for game_number in range(1, NUM_GAMES + 1):
                starter = "A" if game_number % 2 else "B"
                print(f"[{game_number}/{NUM_GAMES}] {starter} starts...", flush=True)
                result = run_game(game_number)
                results.append(result)
                writer.writerow(result)
                output.flush()
                write_summary(summary_path, results)
                outcome = result["winner"].upper() or result["status"]
                print(
                    f"  {outcome}: {result['end_reason']}; "
                    f"captures A/B={result['a_captures']}/{result['b_captures']}; "
                    f"{result['half_moves']} actions; {result['duration_seconds']:.1f}s",
                    flush=True,
                )
                if result["error"]:
                    print(f"  {result['error']}", flush=True)
        except KeyboardInterrupt:
            print("\nStopped. Previously finished games have been saved.")

    write_summary(summary_path, results)
    for row in summarize(results):
        rate = f"{row['win_pct']}%" if row["valid_games"] else "n/a"
        print(
            f"{row['contender'].upper()} ({row['difficulty']}): "
            f"{row['wins']} wins, {row['losses']} losses, {row['draws']} draws; "
            f"win rate {rate} across {row['valid_games']} valid games"
        )
    print(f"Games:   {games_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
