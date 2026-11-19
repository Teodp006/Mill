# Mill — Intelligent Systems Lab

Nine Men’s Morris (Mill) with a Monte Carlo Tree Search agent, playable against a human, against another AI, or in headless batch simulations.

Built on top of [`famnit-gym`](https://github.com/DomenSoberlFamnit/famnit-gym).

---

## Quick start

```bash
source <env_name>/bin/activate
python main.py
```

The program does not need any argument, just make sure that the files `main.py`, `game.py`, `agent.py`, `human_player.py` and `match_stats.py`  as all off them are neaded to execute the code. The responsabilities of each file are detailed down:


---

## Project layout

| File | Role |
|------|------|
| `main.py` | Entry point. Menu, match setup, watch loop and batch simulations. |
| `game.py` | Match loop over the PettingZoo environment. Asks each seat for a move and records timing. |
| `agent.py` | AI player. Monte Carlo Tree Search with four difficulties. |
| `human_player.py` | Mouse / keyboard interaction on the board window. |
| `match_stats.py` | Statistics helper (treated as an external module). Collects, prints and saves results. |


---

## How to play — menu options

When you launch `main.py` you choose a **mode**:

| Option | Meaning |
|--------|---------|
| **a)** Play against AI | You play with the mouse. The opponent is an AI at the difficulty you pick. Board window opens. Some interactions where improved compared with the original version provided on the examples.|
| **b)** See the AI play against another AI | Two AIs play with the board visible. Useful to watch behaviour. |
| **c)** Match Simulation | Game mode used for testing the models against each other. Works without rendering the environment and asks whether to save useful statistics **before** running so can be done |

### Simulation options **c)**

| Choice | Meaning |
|--------|---------|
| **1)** another AI | Freedom to choose two different dificulties or just the same. This was the mode used for meassuring the experiments results|
| **2)** a dumb player | Performs a random action all the time (`None` in code → `env.step(None)`). Useful for testing that montecarlo is improving the game performance. |

### Who starts

After the mode is set, the program asks which of the two named contenders moves first (`1` or `2`).

By default we asume player-1 is the one starting. If you pick contender 2, the seats are swapped and each `Agent` / `HumanPlayer` updates its internal player index.

> For the experiments all were runned with the same amount of matches started by each player as the policy implemented is agresive (benefits captures and mills over blocks) in order to compare both game styles with our strategy agent.

### Difficulties

| Label | Idea | MCTS iterations |
|-------|------|-----------------|
| **base** | Plain Monte Carlo: uniform random expansion and playouts. No tactical root filter. | 1000 |
| **easy**/**medium**/**hard** | Tactical move weights + root prune (take mills / block threats). |20/100/1000|


---

## Statistics and output files

Handled by `match_stats.py`.

**Per match**

- Contenders (`human`, `dumb`, or a difficulty name)
- Who started
- Winner and end reason (`pieces`, `immobilization`, `quit`, `truncation`)
- `material\_margin\_percent` for each seat:
  
    $\frac{(\texttt{own remaining} − \texttt{opponent remaining})}{\texttt{total pieces} \times 100}$  
    - $\texttt{remaining}$: pieces in hand +   pieces on the board
    - $\texttt{total pices}$ = 9
- `mean_decision_ms` only for agent players to meassure the computational time

**Overall session**

- Mean off all the matches played in the same session, if the program is closed an the opened again with same settings it will save the results on another different file.

> Even if the program if the enviroment is closed the program asks to save the stats from the previous matches.

**Where files go**

```
benchmark_results/
  <YYYYMMDD_HHMMSS_microseconds>_games.csv
  <YYYYMMDD_HHMMSS_microseconds>_summary.csv
```


---

## Reproducibility (batch mode)

Each simulated match sets a deterministic seed:

```text
seed = 2026 + (game_number - 1)
```

So game 1 uses `2026`, game 2 uses `2027`, and so on. Both Python’s `random` and NumPy are seeded, and the environment is reset with that seed. Repeating the same batch size and starter should replay the same search noise for a given code version.

---

## Dependencies

- Python 3.12+ recommended
- `famnit-gym` (pulls in Gymnasium, PettingZoo, pygame-ce, NumPy, …)

---

