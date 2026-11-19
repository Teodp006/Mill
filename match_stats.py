"""Match statistics helper for the Mill lab.

This whole module was written by AI. We wanted a way to collect and save a few
simple numbers after each match, but that was not part of the core game work, so
we treated this file as an external helper and left the agents and the match
loop in our own code.

What it tracks
--------------
- Who sat in each seat (human, dumb, or an AI difficulty such as easy / base).
- Who moved first.
- Who won and why the match ended.
- material_margin_percent: how many pieces each side still has, written as a
  percentage of the starting 9. Pieces still in hand count as well.
- mean_decision_ms: average time the AI spent thinking on its turns. Humans and
  the random player leave this blank.

When the user agrees to save, two CSV files are written under benchmark_results/
with a timestamp in the name: one row per match, and one overall row per
difficulty that appeared in the session.
"""

import csv
from datetime import datetime
from pathlib import Path

from human_player import HumanPlayer

# Each player starts with 9 pieces. Unplaced pieces still count as material.
INITIAL_PIECES = 9

# One timestamped pair of files is written here when the user agrees to save.
RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"


def difficulty_label(player):
    """Return the short name we store for a seat in the statistics files.

    Parameters
    ----------
    player :
        The object that sits in that seat. It can be an Agent, a HumanPlayer,
        or None for the random "dumb" player.

    Returns
    -------
    str
        "dumb" for the random player, "human" for a person, or the AI difficulty
        string such as "easy", "medium", "hard" or "base".
    """
    if player is None:
        return "dumb"
    if isinstance(player, HumanPlayer):
        return "human"
    return player.dificulty


def pieces_remaining(model, player):
    """Count how many pieces a player still owns.

    This is the sum of pieces still waiting to be placed and pieces already on
    the board. A capture only lowers this number when a piece leaves the board.

    Parameters
    ----------
    model :
        A MillModel (or a clone of it) that holds the current board state.
    player : int
        1 or 2.

    Returns
    -------
    int
        Total pieces that player still has.
    """
    info = model._player[player]
    return info["pieces_holding"] + info["pieces_playing"]


def material_margin_percent(model, player):
    """Measure how far ahead a player is in pieces, as a percentage of 9.

    The formula is (own remaining - opponent remaining) / 9 * 100. A positive
    value means this player has more pieces left. A win by immobilization can
    still leave a negative margin if the winner had fewer pieces.

    Parameters
    ----------
    model :
        A MillModel with the final (or current) position.
    player : int
        1 or 2.

    Returns
    -------
    float
        Rounded to three decimal places. For example, 4 vs 3 pieces is about
        11.111, and a two-piece lead is about 22.222.
    """
    own = pieces_remaining(model, player)
    opponent = pieces_remaining(model, _opponent(player))
    return round((own - opponent) / INITIAL_PIECES * 100, 3)


def print_match_result(record):
    """Print the short one-line summary used after a single match ends.

    This is the line main.py used to build by hand. Keeping it here means the
    menu file does not need to know which fields exist on a record.

    Parameters
    ----------
    record : dict
        The dictionary returned by GameRecorder.finish (usually after
        MatchSession.add has also set the "game" number).
    """
    print(
        f"Result: winner {record['winner'] or '-'} "
        f"({record['end_reason']}), "
        f"material_margin_percent "
        f"{record['player_1_material_margin_percent']} / "
        f"{record['player_2_material_margin_percent']}"
    )


def close_session(session, save=None):
    """Show the session summary and decide whether to write the CSV files.

    Call this when the user leaves the watch loop, when a batch finishes, or
    when the program is interrupted. It prints the full table first, then
    either saves, skips saving, or asks.

    Parameters
    ----------
    session : MatchSession
        The session that collected the matches.
    save : bool or None, optional
        True always saves (if there is at least one match).
        False never saves and prints a short message.
        None (the default) asks the user with ask_save.
    """
    session.show()
    if not session.games:
        return None
    if save is True:
        return session.save()
    if save is False:
        print("Statistics were not saved.")
        return None
    return session.ask_save()


def _opponent(player):
    """Return the other seat number (1 -> 2, 2 -> 1)."""
    return 2 if player == 1 else 1


def _mean_decision_ms(seconds, moves):
    """Turn total thinking time into an average in milliseconds.

    Parameters
    ----------
    seconds : float
        Sum of all timed AI decisions for that seat.
    moves : int
        How many AI decisions were timed.

    Returns
    -------
    float or str
        The average in milliseconds, or "" when that seat never searched
        (human or dumb player).
    """
    if moves == 0:
        return ""
    return round(seconds * 1000 / moves, 3)


class GameRecorder:
    """Collect the reduced statistics for one single match.

    Create one recorder when the match starts, call add_decision every time an
    AI finishes thinking, then call finish when the match ends. The result is a
    plain dictionary that MatchSession can store and later write to CSV.
    """

    def __init__(self, player_1, player_2):
        """Remember who sat in each seat and who moved first.

        Parameters
        ----------
        player_1 :
            The object in seat 1 after main.py has ordered the players. That
            seat always moves first.
        player_2 :
            The object in seat 2.
        """
        self.difficulties = (
            difficulty_label(player_1),
            difficulty_label(player_2),
        )
        # Player 1 is the seat that moves first after main.py orders the players.
        self.starter = self.difficulties[0]
        self.decision_seconds = [0.0, 0.0]
        self.decision_moves = [0, 0]

    def add_decision(self, player_number, seconds):
        """Record how long one AI turn took.

        Only AI turns should call this. The visual pause after a move, if any,
        should not be included in seconds.

        Parameters
        ----------
        player_number : int
            1 or 2.
        seconds : float
            Wall-clock time spent inside Agent.play for that turn.
        """
        index = player_number - 1
        self.decision_seconds[index] += seconds
        self.decision_moves[index] += 1

    def finish(self, model, end_reason, winner_player):
        """Build the dictionary that describes this finished match.

        Parameters
        ----------
        model :
            The MillModel at the end of the match. Quit and truncation still
            produce a material margin from the last board.
        end_reason : str
            Why the match stopped, for example "pieces", "immobilization",
            "quit" or "truncation".
        winner_player : int or None
            1 or 2 when somebody won. Use None when the match was cut short.

        Returns
        -------
        dict
            Contenders, starter, winner, end reason, both material margins and
            both mean decision times. Internal timing totals are also kept so
            MatchSession can build the overall averages later.
        """
        margins = [
            material_margin_percent(model, 1),
            material_margin_percent(model, 2),
        ]
        if winner_player in (1, 2):
            winner = self.difficulties[winner_player - 1]
        else:
            winner = ""

        return {
            "player_1": self.difficulties[0],
            "player_2": self.difficulties[1],
            "starter": self.starter,
            "winner_player": winner_player or "",
            "winner": winner,
            "end_reason": end_reason,
            "player_1_material_margin_percent": margins[0],
            "player_2_material_margin_percent": margins[1],
            "player_1_mean_decision_ms": _mean_decision_ms(
                self.decision_seconds[0], self.decision_moves[0]
            ),
            "player_2_mean_decision_ms": _mean_decision_ms(
                self.decision_seconds[1], self.decision_moves[1]
            ),
            "decision_seconds": list(self.decision_seconds),
            "decision_moves": list(self.decision_moves),
        }


class MatchSession:
    """Hold every match of one run of the program and build the overall table.

    Add each finished record with add. When the user is done, call show to print
    everything, then ask_save or save. close_session is a small wrapper that
    does both steps for main.py.
    """

    def __init__(self):
        """Start an empty session with no matches yet."""
        self.games = []

    def add(self, record):
        """Append one finished match and give it a 1-based game number.

        Parameters
        ----------
        record : dict
            The dictionary from GameRecorder.finish.
        """
        record["game"] = len(self.games) + 1
        self.games.append(record)

    def summary(self):
        """Build one overall row per difficulty that sat down in this session.

        games_played counts seats, so two players with the same difficulty in
        one match count as two. games_started counts the match once, for the
        difficulty that moved first. A quit or a truncation counts as played
        but not as a win. "base", "human" and "dumb" appear here the same way
        as easy / medium / hard.

        Returns
        -------
        list of dict
            Each row has difficulty, games_played, games_started, wins,
            win_rate, mean_material_margin_percent and mean_decision_ms.
        """
        totals = {}
        for game in self.games:
            # Seat 1 moved first. Count that start once, even if both seats share a difficulty.
            seats = (
                (1, True),
                (2, False),
            )
            for player_number, started in seats:
                difficulty = game[f"player_{player_number}"]
                row = totals.setdefault(difficulty, {
                    "difficulty": difficulty,
                    "games_played": 0,
                    "games_started": 0,
                    "wins": 0,
                    "margin_sum": 0.0,
                    "decision_seconds": 0.0,
                    "decision_moves": 0,
                })
                row["games_played"] += 1
                row["games_started"] += int(started)
                row["wins"] += int(game["winner_player"] == player_number)
                row["margin_sum"] += game[f"player_{player_number}_material_margin_percent"]
                row["decision_seconds"] += game["decision_seconds"][player_number - 1]
                row["decision_moves"] += game["decision_moves"][player_number - 1]

        summary = []
        for row in totals.values():
            played = row["games_played"]
            summary.append({
                "difficulty": row["difficulty"],
                "games_played": played,
                "games_started": row["games_started"],
                "wins": row["wins"],
                "win_rate": round(row["wins"] / played * 100, 1) if played else "",
                "mean_material_margin_percent": round(row["margin_sum"] / played, 3) if played else "",
                "mean_decision_ms": _mean_decision_ms(
                    row["decision_seconds"], row["decision_moves"]
                ),
            })
        return summary

    def show(self):
        """Print every match of this session and then the overall table.

        The fields are the same ones that end up in the CSV files. If no match
        was played, a short message is printed instead.
        """
        print("\nMatches")
        if not self.games:
            print("No matches were played.")
            return

        for game in self.games:
            print(
                f"{game['game']}: {game['player_1']} vs {game['player_2']} | "
                f"starter {game['starter']} | "
                f"winner {game['winner'] or '-'} | "
                f"{game['end_reason']} | "
                f"material_margin_percent "
                f"{game['player_1_material_margin_percent']} / "
                f"{game['player_2_material_margin_percent']} | "
                f"mean_decision_ms "
                f"{game['player_1_mean_decision_ms'] or '-'} / "
                f"{game['player_2_mean_decision_ms'] or '-'}"
            )

        print("\nOverall")
        for row in self.summary():
            print(
                f"{row['difficulty']}: "
                f"games_played {row['games_played']}, "
                f"games_started {row['games_started']}, "
                f"win_rate {row['win_rate']}% , "
                f"mean_material_margin_percent {row['mean_material_margin_percent']}, "
                f"mean_decision_ms {row['mean_decision_ms'] or '-'}"
            )

    def ask_save(self):
        """Ask whether to write the CSV files, then save if the answer is yes.

        Returns
        -------
        tuple of Path or None
            The two file paths when the user saves, otherwise None.
        """
        while True:
            answer = input("Save the statistics? (y/n): ").strip().lower()
            if answer in ("y", "yes", "s", "si", "sí"):
                return self.save()
            if answer in ("n", "no"):
                print("Statistics were not saved.")
                return None
            print("Choose y or n.")

    def save(self):
        """Write the match list and the overall table into benchmark_results/.

        File names use the current date and time, the same idea as the old
        benchmark script.

        Returns
        -------
        tuple of Path
            (games_csv_path, summary_csv_path).
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        games_path = RESULTS_DIR / f"{stamp}_games.csv"
        summary_path = RESULTS_DIR / f"{stamp}_summary.csv"

        game_fields = [
            "game",
            "player_1",
            "player_2",
            "starter",
            "winner_player",
            "winner",
            "end_reason",
            "player_1_material_margin_percent",
            "player_2_material_margin_percent",
            "player_1_mean_decision_ms",
            "player_2_mean_decision_ms",
        ]
        summary_fields = [
            "difficulty",
            "games_played",
            "games_started",
            "wins",
            "win_rate",
            "mean_material_margin_percent",
            "mean_decision_ms",
        ]

        self._write_csv(games_path, game_fields, self._public_games())
        self._write_csv(summary_path, summary_fields, self.summary())
        print(f"Matches: {games_path}")
        print(f"Overall: {summary_path}")
        return games_path, summary_path

    def _public_games(self):
        """Return match rows without the internal timing totals used by summary."""
        hidden = {"decision_seconds", "decision_moves"}
        return [
            {key: value for key, value in game.items() if key not in hidden}
            for game in self.games
        ]

    def _write_csv(self, path, fieldnames, rows):
        """Write one CSV table with a header row."""
        with path.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
