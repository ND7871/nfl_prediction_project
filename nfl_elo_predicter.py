# nfl_elo_predicter.py
import pandas as pd

# -------------------------------
# Step 1: Team name mapping
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

# -------------------------------
# --- Load and clean the data ---
df = pd.read_csv("nfl_2025_results.csv")

# Keep only the relevant columns
df = df[["Week", "Date", "Winner/tie", "Loser/tie", "PtsW", "PtsL"]]

# Map full names to nicknames
df["team1"] = df["Winner/tie"].map(TEAM_NAME_MAP)
df["team2"] = df["Loser/tie"].map(TEAM_NAME_MAP)
df["score1"] = df["PtsW"]
df["score2"] = df["PtsL"]

# Rename columns to lowercase for consistency
df = df.rename(columns={"Week": "week", "Date": "date"})

# Final cleaned frame
df = df[["week", "date", "team1", "team2", "score1", "score2"]]

# -------------------------------
# Step 3: Initialize Elo ratings
# -------------------------------
teams = {team: 1500 for team in pd.concat([df["team1"], df["team2"]]).unique()}

def expected_score(team_rating, opp_rating):
    """Probability team wins given ratings."""
    return 1 / (1 + 10 ** ((opp_rating - team_rating) / 400))

def update_elo(team_rating, opp_rating, score, K=20):
    """Update Elo rating after a game."""
    exp = expected_score(team_rating, opp_rating)
    return team_rating + K * (score - exp)

def process_game(team1, team2, score1, score2):
    """Update both teams' ratings after a game."""
    if score1 > score2:
        s1, s2 = 1, 0
    elif score1 < score2:
        s1, s2 = 0, 1
    else:
        s1, s2 = 0.5, 0.5
    
    new1 = update_elo(teams[team1], teams[team2], s1)
    new2 = update_elo(teams[team2], teams[team1], s2)
    
    teams[team1], teams[team2] = new1, new2

# -------------------------------
# Step 4: Run through the season
# -------------------------------
for _, row in df.iterrows():
    process_game(row["team1"], row["team2"], int(row["score1"]), int(row["score2"]))

# -------------------------------
# Step 5: Rankings and predictions
# -------------------------------
def power_rankings():
    """Return teams sorted by Elo rating."""
    return sorted(teams.items(), key=lambda x: x[1], reverse=True)

def predict_game(team1, team2):
    """Predict outcome probabilities for a matchup."""
    prob = expected_score(teams[team1], teams[team2])
    return {
        "team1": team1,
        "team2": team2,
        "prob_team1_win": round(prob, 3),
        "prob_team2_win": round(1 - prob, 3)
    }

# -------------------------------
# Step 6: Example usage
# -------------------------------
if __name__ == "__main__":
    print("NFL Elo Power Rankings (2025):")
    for rank, (team, rating) in enumerate(power_rankings(), 1):
        print(f"{rank}. {team} ({rating:.1f})")

    print("\nExample Prediction:")
    print(predict_game("Bengals", "Ravens"))