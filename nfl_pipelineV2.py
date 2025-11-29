import requests
import pandas as pd
import numpy as np
import datetime
from time import sleep
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
import joblib
import argparse

# --- Config ---
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
START_SEASON = 2021
REGULAR_SEASON_WEEKS = 18
SLEEP_BETWEEN_REQUESTS = 0.2
TIMEOUT = 15

MODEL_FILE = "nfl_win_predictor.pkl"
ENCODERS_FILE = "encoders.pkl"
ENGINEERED_FILE = "engineered_nfl_features.csv"

# Choose model: "rf" or "logreg"
DEFAULT_MODEL = "logreg"

# Utilities
def get_current_season() -> int:
    today = datetime.date.today()
    return today.year

def pull_scoreboard(season: int, week: int) -> Dict[str, Any]:
    params = {"year": season, "week": week}
    resp = requests.get(SCOREBOARD_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def parse_event_to_game(event: Dict[str, Any], season: int, week: int) -> Optional[Dict[str, Any]]:
    try:
        event_id = event.get("id")
        comp = event["competitions"][0]
        competitors = comp["competitors"]
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        game = {
            "event_id": event_id,
            "season": season,
            "week": week,
            "date": event.get("date"),
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": int(home.get("score", 0)),
            "away_score": int(away.get("score", 0)),
            "venue": comp.get("venue", {}).get("fullName", "Generic Venue")
        }
        return game
    except Exception:
        return None

def pull_summary(event_id: str) -> Dict[str, Any]:
    resp = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()

# --- Robust extraction (players-first, team-total fallback) ---
def extract_team_total(summary: Dict[str, Any], team_name: str, stat_name: str, default: int = 0) -> int:
    teams = summary.get("boxscore", {}).get("teams", [])
    for t in teams:
        if t.get("team", {}).get("displayName") != team_name:
            continue
        for stat in t.get("statistics", []):
            if stat.get("name") == stat_name:
                val = stat.get("displayValue")
                try:
                    # handles "23/36", "1-6", "8-69", etc.
                    return int(str(val).replace(",", " ").split()[0].split("/")[0].split("-")[0])
                except Exception:
                    return default
    return default

def extract_team_stat_players(summary: Dict[str, Any], team_name: str, category: str, label: str) -> int:
    players = summary.get("boxscore", {}).get("players", [])
    for team in players:
        if team.get("team", {}).get("displayName") != team_name:
            continue
        for group in team.get("statistics", []):
            if (group.get("name") or "").lower() == category.lower():
                labels = group.get("labels", []) or []
                totals = group.get("totals", []) or []
                if label in labels:
                    idx = labels.index(label)
                    try:
                        return int(str(totals[idx]).split()[0])
                    except Exception:
                        return 0
    return 0

def extract_turnovers(summary: Dict[str, Any], team_name: str) -> int:
    return extract_team_total(summary, team_name, "turnovers", default=0)

def enrich_game_with_player_stats(game: dict) -> dict:
    event_id = game["event_id"]
    try:
        summary = pull_summary(event_id)
        sleep(SLEEP_BETWEEN_REQUESTS)

        # Players-first yards
        home_pass = extract_team_stat_players(summary, game["home_team"], "passing", "YDS")
        away_pass = extract_team_stat_players(summary, game["away_team"], "passing", "YDS")
        home_rush = extract_team_stat_players(summary, game["home_team"], "rushing", "YDS")
        away_rush = extract_team_stat_players(summary, game["away_team"], "rushing", "YDS")
        home_recv = extract_team_stat_players(summary, game["home_team"], "receiving", "YDS")
        away_recv = extract_team_stat_players(summary, game["away_team"], "receiving", "YDS")

        # Fallback to team totals if player blocks missing
        if home_pass == 0:
            home_pass = extract_team_total(summary, game["home_team"], "netPassingYards", default=0)
        if away_pass == 0:
            away_pass = extract_team_total(summary, game["away_team"], "netPassingYards", default=0)
        if home_rush == 0:
            home_rush = extract_team_total(summary, game["home_team"], "rushingYards", default=0)
        if away_rush == 0:
            away_rush = extract_team_total(summary, game["away_team"], "rushingYards", default=0)
        # Receiving totals may be absent; approximate with netPassingYards if missing
        if home_recv == 0:
            home_recv = home_pass
        if away_recv == 0:
            away_recv = away_pass

        home_to = extract_turnovers(summary, game["home_team"])
        away_to = extract_turnovers(summary, game["away_team"])

        game.update({
            "home_qb_passing_yards": home_pass,
            "away_qb_passing_yards": away_pass,
            "home_rb_rushing_yards": home_rush,
            "away_rb_rushing_yards": away_rush,
            "home_wr_receiving_yards": home_recv,
            "away_wr_receiving_yards": away_recv,
            "home_turnovers": home_to,
            "away_turnovers": away_to,
            "summary_ok": 1
        })
    except Exception as e:
        print(f"-> Failed to enrich {event_id}: {e}")
        for col in [
            "home_qb_passing_yards","away_qb_passing_yards",
            "home_rb_rushing_yards","away_rb_rushing_yards",
            "home_wr_receiving_yards","away_wr_receiving_yards",
            "home_turnovers","away_turnovers"
        ]:
            game[col] = 0
        game["summary_ok"] = 0
    return game

# Pull orchestration with week logging
def pull_season_week(season: int, week: int) -> pd.DataFrame:
    try:
        raw = pull_scoreboard(season, week)
        events = raw.get("events", [])
        if not events:
            print(f"   ✗ No events for season {season}, week {week}")
            return pd.DataFrame()

        parsed_games: List[Dict[str, Any]] = []
        for event in events:
            base = parse_event_to_game(event, season, week)
            if base is None:
                continue
            enriched = enrich_game_with_player_stats(base)
            parsed_games.append(enriched)

        print(f"   ✓ Retrieved {len(parsed_games)} games for season {season}, week {week}")
        return pd.DataFrame(parsed_games)
    except Exception as e:
        print(f"   ! Error pulling season {season}, week {week}: {e}")
        return pd.DataFrame()

def pull_all_data(start_season: int = START_SEASON, end_season: Optional[int] = None) -> pd.DataFrame:
    if end_season is None:
        end_season = get_current_season()
    all_rows: List[pd.DataFrame] = []
    for season in range(start_season, end_season + 1):
        for week in range(1, REGULAR_SEASON_WEEKS + 1):
            print(f"-> Pulling season {season}, week {week}...")
            df_week = pull_season_week(season, week)
            if not df_week.empty:
                all_rows.append(df_week)
            sleep(SLEEP_BETWEEN_REQUESTS)
    if not all_rows:
        return pd.DataFrame()
    full_df = pd.concat(all_rows, ignore_index=True)
    print(f"-> Total games pulled: {len(full_df)} from seasons {start_season}-{end_season}.")
    return full_df

# Feature engineering
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("-> Engineering features...")

    # Outcomes
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["away_win"] = (df["away_score"] > df["home_score"]).astype(int)

    # Ensure numeric scores before differential
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)

    # Score differential (correctly computed)
    df["home_score_diff"] = df["home_score"] - df["away_score"]
    df["away_score_diff"] = df["away_score"] - df["home_score"]

    # Ensure numeric player stats
    for col in [
        "home_qb_passing_yards","away_qb_passing_yards",
        "home_rb_rushing_yards","away_rb_rushing_yards",
        "home_wr_receiving_yards","away_wr_receiving_yards",
        "home_turnovers","away_turnovers"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Rolling averages (3-game window, within season)
    def rolling(team_col: str, value_col: str, window: int = 3):
        return df.groupby(["season", team_col])[value_col].transform(lambda x: x.rolling(window, min_periods=1).mean())

    def ewm(team_col: str, value_col: str, span: int = 3):
        return df.groupby(["season", team_col])[value_col].transform(lambda x: x.ewm(span=span).mean())

    # Team points and recent form
    df["home_points_rolling"] = rolling("home_team", "home_score")
    df["away_points_rolling"] = rolling("away_team", "away_score")
    df["home_points_recent"] = ewm("home_team", "home_score")
    df["away_points_recent"] = ewm("away_team", "away_score")

    # Rolling score differential
    df["home_score_diff_rolling"] = rolling("home_team", "home_score_diff")
    df["away_score_diff_rolling"] = rolling("away_team", "away_score_diff")

    # Player stats rolling
    df["home_qb_pass_rolling"] = rolling("home_team", "home_qb_passing_yards")
    df["away_qb_pass_rolling"] = rolling("away_team", "away_qb_passing_yards")
    df["home_rb_rush_rolling"] = rolling("home_team", "home_rb_rushing_yards")
    df["away_rb_rush_rolling"] = rolling("away_team", "away_rb_rushing_yards")
    df["home_wr_recv_rolling"] = rolling("home_team", "home_wr_receiving_yards")
    df["away_wr_recv_rolling"] = rolling("away_team", "away_wr_receiving_yards")
    df["home_turnovers_rolling"] = rolling("home_team", "home_turnovers")
    df["away_turnovers_rolling"] = rolling("away_team", "away_turnovers")

    # Season record win pct (expanding within season, up to current row)
    df["home_season_win_pct"] = df.groupby(["season", "home_team"])["home_win"].transform(lambda x: x.expanding().mean())
    df["away_season_win_pct"] = df.groupby(["season", "away_team"])["away_win"].transform(lambda x: x.expanding().mean())

    # Opponent strength (opponent's expanding win pct within season)
    df["home_opponent_win_pct"] = df.groupby(["season", "away_team"])["away_win"].transform(lambda x: x.expanding().mean())
    df["away_opponent_win_pct"] = df.groupby(["season", "home_team"])["home_win"].transform(lambda x: x.expanding().mean())

    # Weeks into season
    df["weeks_into_season"] = df["week"]
    df["is_home"] = 1  # used at inference, mostly constant for row construction

    print("-> Feature engineering complete.")
    return df

# Data preparation
FEATURE_COLUMNS = [
    # identity and time
    "home_team_encoded","away_team_encoded","week","season","is_home","weeks_into_season",
    # team form & recent performance
    "home_points_rolling","away_points_rolling","home_points_recent","away_points_recent",
    "home_score_diff_rolling","away_score_diff_rolling",
    # records and opponent strength
    "home_season_win_pct","away_season_win_pct",
    "home_opponent_win_pct","away_opponent_win_pct",
    # player performance
    "home_qb_pass_rolling","away_qb_pass_rolling",
    "home_rb_rush_rolling","away_rb_rush_rolling",
    "home_wr_recv_rolling","away_wr_recv_rolling",
    # ball security
    "home_turnovers_rolling","away_turnovers_rolling",
]

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    print("-> Preparing data for training...")
    # Drop games missing scores
    df = df.dropna(subset=["home_score", "away_score"])
    y = df["home_win"]

    # Encode teams
    team_encoder = LabelEncoder()
    team_encoder.fit(pd.concat([df["home_team"], df["away_team"]]).unique())
    df["home_team_encoded"] = team_encoder.transform(df["home_team"])
    df["away_team_encoded"] = team_encoder.transform(df["away_team"])

    X = df[FEATURE_COLUMNS].fillna(0)
    print(f"-> Prepared {X.shape[0]} samples with {X.shape[1]} features.")
    return X, y, team_encoder

# Modeling
def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = DEFAULT_MODEL):
    print(f"-> Starting model training ({model_type})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "logreg":
        # Logistic Regression baseline (probability-focused)
        model = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        print(f"-> Accuracy: {acc:.3f} | LogLoss: {ll:.3f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        return model

    # RandomForest with grid search
    param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [None, 12, 24],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2", None],
    }

    print("-> Running GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model: RandomForestClassifier = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"-> Accuracy: {acc:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return best_model

# Main
def main(start_season: int, end_season: Optional[int], model_type: str):
    print("=== NFL Pipeline: Pull + Enrich + Engineer + Train ===")
    if end_season is None:
        end_season = get_current_season()
    print(f"-> Pulling seasons {start_season} through {end_season} (weeks 1-{REGULAR_SEASON_WEEKS})...")
    raw_df = pull_all_data(start_season=start_season, end_season=end_season)

    if raw_df.empty:
        print("-> No data pulled. Exiting.")
        raise SystemExit(1)

    print(f"-> Raw rows: {len(raw_df)}")
    engineered_df = add_features(raw_df)
    engineered_df.to_csv(ENGINEERED_FILE, index=False)
    print(f"-> Engineered dataset saved to {ENGINEERED_FILE}")

    X, y, team_encoder = prepare_data(engineered_df)
    model = train_model(X, y, model_type=model_type)

    joblib.dump(model, MODEL_FILE)
    joblib.dump({"team_encoder": team_encoder, "feature_columns": FEATURE_COLUMNS}, ENCODERS_FILE)
    print(f"-> Model saved to {MODEL_FILE}")
    print(f"-> Encoders saved to {ENCODERS_FILE}")
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL Pipeline")
    parser.add_argument("--start-season", type=int, default=START_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--model", type=str, choices=["rf","logreg"], default=DEFAULT_MODEL)
    args = parser.parse_args()

    main(start_season=args.start_season, end_season=args.end_season, model_type=args.model)