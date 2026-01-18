#!/usr/bin/env python3
"""Train a model to predict next-game performance after a bad game."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class FeatureConfig:
    target: str
    rolling_window: int
    min_minutes: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a model to predict next-game PTS or PRA after bad games. "
            "Bad games are below both season and rolling averages."
        )
    )
    parser.add_argument("--data", required=True, help="Path to a CSV of game logs.")
    parser.add_argument(
        "--target",
        choices=["PTS", "PRA"],
        default="PTS",
        help="Target to predict for the next game.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window size for recent form features.",
    )
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=20.0,
        help="Minimum minutes played to keep a game for training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for the test split (time-based).",
    )
    parser.add_argument(
        "--output-model",
        default="model.joblib",
        help="Output path for the trained model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        default="metrics.json",
        help="Output path for the evaluation metrics JSON.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, target: str) -> None:
    required = {
        "player_id",
        "game_date",
        "minutes",
        "opponent_def_rating",
        "is_star",
    }
    required.add(target.lower())
    missing = required - set(df.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_sorted}")


def _prepare_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    target_col = config.target.lower()
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    def add_group_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        shifted_target = group[target_col].shift(1)
        group["season_avg"] = shifted_target.expanding().mean()
        group["rolling_avg"] = shifted_target.rolling(config.rolling_window).mean()
        group["last_game"] = shifted_target
        group["next_game"] = group[target_col].shift(-1)
        group["minutes_last_game"] = group["minutes"].shift(1)
        group["opponent_def_rating"] = group["opponent_def_rating"].astype(float)
        group["is_star"] = group["is_star"].astype(int)
        return group

    df = df.groupby("player_id", group_keys=False).apply(add_group_features)

    df["is_bad_game"] = (
        (df[target_col] < df["season_avg"]) & (df[target_col] < df["rolling_avg"])
    )

    df = df[df["minutes"] >= config.min_minutes]
    df = df[df["minutes_last_game"] >= config.min_minutes]
    df = df[df["is_bad_game"]]
    df = df.dropna(subset=["season_avg", "rolling_avg", "last_game", "next_game"])

    feature_cols = [
        "season_avg",
        "rolling_avg",
        "last_game",
        "opponent_def_rating",
        "is_star",
        "minutes",
        "minutes_last_game",
    ]

    return df[["player_id", "game_date", "next_game", *feature_cols]]


def _time_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Not enough data to perform the requested split.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _train_model(
    train_df: pd.DataFrame, feature_cols: Iterable[str]
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(random_state=42)
    model.fit(train_df[list(feature_cols)], train_df["next_game"])
    return model


def _evaluate(
    model: GradientBoostingRegressor,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> dict:
    preds = model.predict(test_df[list(feature_cols)])
    mae = mean_absolute_error(test_df["next_game"], preds)
    rmse = mean_squared_error(test_df["next_game"], preds, squared=False)
    r2 = r2_score(test_df["next_game"], preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    args = parse_args()
    target = args.target.upper()
    config = FeatureConfig(
        target=target,
        rolling_window=args.rolling_window,
        min_minutes=args.min_minutes,
    )

    df = pd.read_csv(args.data)
    _validate_columns(df, target)

    features_df = _prepare_features(df, config)

    train_df, test_df = _time_split(features_df, args.test_size)
    feature_cols = [
        "season_avg",
        "rolling_avg",
        "last_game",
        "opponent_def_rating",
        "is_star",
        "minutes",
        "minutes_last_game",
    ]

    model = _train_model(train_df, feature_cols)
    metrics = _evaluate(model, test_df, feature_cols)

    joblib.dump({"model": model, "features": feature_cols, "target": target}, args.output_model)
    pd.Series(metrics).to_json(args.metrics_out, indent=2)

    print("Training complete.")
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
