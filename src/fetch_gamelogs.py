#!/usr/bin/env python3
"""Fetch NBA player game logs via nba_api and save to CSV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, playergamelog
from nba_api.stats.static import players


@dataclass
class FetchConfig:
    season: str
    season_type: str
    star_ids: set[int]
    max_players: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NBA game logs into a CSV.")
    parser.add_argument(
        "--season",
        default="2023-24",
        help="Season string, e.g. 2023-24.",
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help="Season type string, e.g. Regular Season or Playoffs.",
    )
    parser.add_argument(
        "--player-ids",
        default="",
        help="Comma-separated player IDs to download. Defaults to active players.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Optional cap on number of players to fetch.",
    )
    parser.add_argument(
        "--star-ids",
        default="",
        help="Comma-separated player IDs to flag as stars.",
    )
    parser.add_argument(
        "--output",
        default="gamelogs.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def _parse_minutes(value: object) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if ":" not in text:
        return float(text)
    minutes_str, seconds_str = text.split(":", maxsplit=1)
    return float(minutes_str) + float(seconds_str) / 60.0


def _parse_player_ids(raw: str, default_ids: List[int]) -> List[int]:
    if not raw:
        return default_ids
    return [int(item) for item in raw.split(",") if item.strip()]


def _load_opponent_def_rating(season: str, season_type: str) -> dict[str, float]:
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    return dict(zip(team_stats["TEAM_ABBREVIATION"], team_stats["DEF_RATING"]))


def _extract_opponent(matchup: str) -> str:
    parts = matchup.strip().split()
    return parts[-1] if parts else ""


def _fetch_player_logs(player_id: int, season: str, season_type: str) -> pd.DataFrame:
    return playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    ).get_data_frames()[0]


def _normalize_logs(
    raw_logs: pd.DataFrame,
    player_id: int,
    def_rating_map: dict[str, float],
    star_ids: set[int],
) -> pd.DataFrame:
    logs = raw_logs.copy()
    logs["player_id"] = player_id
    logs["game_date"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
    logs["pts"] = logs["PTS"].astype(float)
    logs["reb"] = logs["REB"].astype(float)
    logs["ast"] = logs["AST"].astype(float)
    logs["pra"] = logs["pts"] + logs["reb"] + logs["ast"]
    logs["minutes"] = logs["MIN"].apply(_parse_minutes)
    logs["opponent"] = logs["MATCHUP"].map(_extract_opponent)
    logs["opponent_def_rating"] = logs["opponent"].map(def_rating_map)
    logs["is_star"] = 1 if player_id in star_ids else 0
    return logs[
        [
            "player_id",
            "game_date",
            "pts",
            "pra",
            "minutes",
            "opponent_def_rating",
            "is_star",
        ]
    ]


def _limit_players(player_ids: Iterable[int], max_players: int | None) -> List[int]:
    player_list = list(player_ids)
    if max_players is None:
        return player_list
    return player_list[:max_players]


def main() -> None:
    args = parse_args()
    star_ids = {int(item) for item in args.star_ids.split(",") if item.strip()}
    config = FetchConfig(
        season=args.season,
        season_type=args.season_type,
        star_ids=star_ids,
        max_players=args.max_players,
    )

    active_players = players.get_active_players()
    default_ids = [player["id"] for player in active_players]
    player_ids = _parse_player_ids(args.player_ids, default_ids)
    player_ids = _limit_players(player_ids, config.max_players)

    def_rating_map = _load_opponent_def_rating(config.season, config.season_type)

    all_logs = []
    for player_id in player_ids:
        raw_logs = _fetch_player_logs(player_id, config.season, config.season_type)
        if raw_logs.empty:
            continue
        all_logs.append(
            _normalize_logs(raw_logs, player_id, def_rating_map, config.star_ids)
        )

    if not all_logs:
        raise ValueError("No game logs were downloaded.")

    combined = pd.concat(all_logs, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"Saved {len(combined):,} rows to {args.output}.")


if __name__ == "__main__":
    main()
