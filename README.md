# nba-bad-game-effects

Detecting “awful” games and predicting next-game performance for PTS or PRA.

## Overview
This starter workflow trains a simple model to estimate next-game performance
when a player has an especially poor game relative to both their season average
and recent rolling average. It filters out games where the player did not log
meaningful minutes (e.g., injury or blowout scenarios) before training.

## Data format
Provide a CSV of game logs with at least the following columns:

| Column | Description |
| --- | --- |
| `player_id` | Player identifier (string or numeric). |
| `game_date` | Game date in a parseable format (YYYY-MM-DD recommended). |
| `pts` | Points scored in the game. |
| `pra` | Points + rebounds + assists in the game. |
| `minutes` | Minutes played in the game. |
| `opponent_def_rating` | Opponent defensive rating (lower = stronger defense). |
| `is_star` | 1 if the player is considered a star, else 0. |

## Fetching data with nba_api
You can pull a CSV from the public NBA stats endpoints using the
[`nba_api`](https://github.com/swar/nba_api) package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_gamelogs.py --season 2023-24 --output gamelogs.csv
```

Helpful options:

- `--player-ids` (comma-separated) to limit downloads to specific players.
- `--max-players` to cap the number of active players fetched.
- `--star-ids` (comma-separated) to flag stars in the output.
- `--season-type` to switch to `Playoffs` if needed.

## Training
Train the model once you have a CSV:

```bash
python src/train_model.py --data gamelogs.csv --target PTS
```

Optional flags:

- `--target` (`PTS` or `PRA`)
- `--rolling-window` (default: 5)
- `--min-minutes` (default: 20)
- `--test-size` (default: 0.2)
- `--output-model` (default: `model.joblib`)
- `--metrics-out` (default: `metrics.json`)

## How it works
The script builds features per player:

- Season average of the target prior to each game.
- Rolling average of the target over the last N games.
- Last game performance.
- Opponent defensive rating.
- Star indicator.
- Minutes played in current and prior games.

A game is labeled as a **bad game** if the target is below both the season and
rolling averages. The model trains only on bad-game samples that meet the
minimum minutes threshold and predicts the *next* game result for the player.
