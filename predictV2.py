import pandas as pd
import joblib

MODEL_FILE = "nfl_win_predictor.pkl"
ENCODERS_FILE = "encoders.pkl"
DATA_FILE = "engineered_nfl_features.csv"

# -------------------------------
# Team nickname mapping
# -------------------------------
TEAM_NAME_MAP = {
    "Arizona Cardinals": "Cardinals",
    "Atlanta Falcons": "Falcons",
    "Baltimore Ravens": "Ravens",
    "Buffalo Bills": "Bills",
    "Carolina Panthers": "Panthers",
    "Chicago Bears": "Bears",
    "Cincinnati Bengals": "Bengals",
    "Cleveland Browns": "Browns",
    "Dallas Cowboys": "Cowboys",
    "Denver Broncos": "Broncos",
    "Detroit Lions": "Lions",
    "Green Bay Packers": "Packers",
    "Houston Texans": "Texans",
    "Indianapolis Colts": "Colts",
    "Jacksonville Jaguars": "Jaguars",
    "Kansas City Chiefs": "Chiefs",
    "Las Vegas Raiders": "Raiders",
    "Los Angeles Chargers": "Chargers",
    "Los Angeles Rams": "Rams",
    "Miami Dolphins": "Dolphins",
    "Minnesota Vikings": "Vikings",
    "New England Patriots": "Patriots",
    "New Orleans Saints": "Saints",
    "New York Giants": "Giants",
    "New York Jets": "Jets",
    "Philadelphia Eagles": "Eagles",
    "Pittsburgh Steelers": "Steelers",
    "San Francisco 49ers": "49ers",
    "Seattle Seahawks": "Seahawks",
    "Tampa Bay Buccaneers": "Buccaneers",
    "Tennessee Titans": "Titans",
    "Washington Commanders": "Commanders"
}

def normalize_team(user_input: str) -> str:
    """Map nickname or full name to canonical full name used in encoder."""
    s = user_input.strip().lower()
    # Try nickname match
    for full, nick in TEAM_NAME_MAP.items():
        if nick.lower() == s:
            return full
    # Try full name match
    for full in TEAM_NAME_MAP.keys():
        if full.lower() == s:
            return full
    raise ValueError(f"Unrecognized team name: {user_input}")

# -------------------------------
# Model and feature utilities
# -------------------------------
def load_model_and_encoders():
    model = joblib.load(MODEL_FILE)
    encoders = joblib.load(ENCODERS_FILE)
    print("-> Model and encoders loaded.")
    return model, encoders

def team_subset(df, team, season, week, prefix):
    if prefix == "home":
        return df[(df["season"] == season) & (df["home_team"] == team) & (df["week"] < week)].copy()
    return df[(df["season"] == season) & (df["away_team"] == team) & (df["week"] < week)].copy()

def last_or_zero(series):
    return float(series.iloc[-1]) if not series.empty else 0.0

def build_features_row(df, team_encoder, home_team, away_team, season, week, feature_columns):
    # Encode teams
    home_team_encoded = team_encoder.transform([home_team])[0]
    away_team_encoded = team_encoder.transform([away_team])[0]

    # Subsets
    home_games = team_subset(df, home_team, season, week, "home")
    away_games = team_subset(df, away_team, season, week, "away")

    # Rolling stats
    home_points_rolling = last_or_zero(home_games["home_points_rolling"])
    away_points_rolling = last_or_zero(away_games["away_points_rolling"])
    home_points_recent = last_or_zero(home_games["home_points_recent"])
    away_points_recent = last_or_zero(away_games["away_points_recent"])
    home_score_diff_rolling = last_or_zero(home_games["home_score_diff_rolling"])
    away_score_diff_rolling = last_or_zero(away_games["away_score_diff_rolling"])

    # Season/opponent strength
    home_season_win_pct = last_or_zero(home_games["home_season_win_pct"])
    away_season_win_pct = last_or_zero(away_games["away_season_win_pct"])
    home_opponent_win_pct = last_or_zero(home_games["home_opponent_win_pct"])
    away_opponent_win_pct = last_or_zero(away_games["away_opponent_win_pct"])

    # Player performance
    home_qb_pass_rolling = last_or_zero(home_games["home_qb_pass_rolling"])
    away_qb_pass_rolling = last_or_zero(away_games["away_qb_pass_rolling"])
    home_rb_rush_rolling = last_or_zero(home_games["home_rb_rush_rolling"])
    away_rb_rush_rolling = last_or_zero(away_games["away_rb_rush_rolling"])
    home_wr_recv_rolling = last_or_zero(home_games["home_wr_recv_rolling"])
    away_wr_recv_rolling = last_or_zero(away_games["away_wr_recv_rolling"])

    # Ball security
    home_turnovers_rolling = last_or_zero(home_games["home_turnovers_rolling"])
    away_turnovers_rolling = last_or_zero(away_games["away_turnovers_rolling"])

    features = pd.DataFrame([{
        "home_team_encoded": home_team_encoded,
        "away_team_encoded": away_team_encoded,
        "week": week,
        "season": season,
        "is_home": 1,
        "weeks_into_season": week,
        "home_points_rolling": home_points_rolling,
        "away_points_rolling": away_points_rolling,
        "home_points_recent": home_points_recent,
        "away_points_recent": away_points_recent,
        "home_score_diff_rolling": home_score_diff_rolling,
        "away_score_diff_rolling": away_score_diff_rolling,
        "home_season_win_pct": home_season_win_pct,
        "away_season_win_pct": away_season_win_pct,
        "home_opponent_win_pct": home_opponent_win_pct,
        "away_opponent_win_pct": away_opponent_win_pct,
        "home_qb_pass_rolling": home_qb_pass_rolling,
        "away_qb_pass_rolling": away_qb_pass_rolling,
        "home_rb_rush_rolling": home_rb_rush_rolling,
        "away_rb_rush_rolling": away_rb_rush_rolling,
        "home_wr_recv_rolling": home_wr_recv_rolling,
        "away_wr_recv_rolling": away_wr_recv_rolling,
        "home_turnovers_rolling": home_turnovers_rolling,
        "away_turnovers_rolling": away_turnovers_rolling
    }])

    return features[feature_columns]

def show_feature_importance(model, feature_columns):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
        print("\nFeature Importance Rankings:")
        for feat, imp in order.items():
            print(f"{feat}: {imp:.3f}")
    else:
        print("Model does not support feature importance (likely Logistic Regression).")

# -------------------------------
# Main prediction
# -------------------------------
def main(home_team: str, away_team: str, season: int, week: int):
    model, encoders = load_model_and_encoders()
    df = pd.read_csv(DATA_FILE)
    team_encoder = encoders["team_encoder"]
    feature_columns = encoders.get("feature_columns")

    # Sanity check
    for t in [home_team, away_team]:
        if t not in team_encoder.classes_:
            raise ValueError(f"Team '{t}' not found in encoder classes. Retrain or adjust teams.")

    features = build_features_row(df, team_encoder, home_team, away_team, season, week, feature_columns)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][1]
    else:
        prob = float(model.predict(features)[0])

    pred = model.predict(features)[0]
    outcome = "Home team wins" if pred == 1 else "Home team loses"

    print(f"\nMatchup: {home_team} vs {away_team} (Week {week}, {season})")
    print(f"Prediction: {outcome} (probability {prob:.2f})")

    show_feature_importance(model, feature_columns)

# -------------------------------
# Interactive mode
# -------------------------------
if __name__ == "__main__":
    print("NFL ML Predictor — interactive mode")

    home_in = input("Enter home team (nickname or full name): ").strip()
    away_in = input("Enter away team (nickname or full name): ").strip()
    season = int(input("Enter season year (e.g. 2025): ").strip())
    week = int(input("Enter week number: ").strip())

    # Normalize to canonical names
    home_full = normalize_team(home_in)
    away_full = normalize_team(away_in)

    main(home_team=home_full, away_team=away_full, season=season, week=week)