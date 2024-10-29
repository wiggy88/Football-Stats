import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

# Load the datasets
data = pd.read_csv("allleagues_backtestdata.csv")
fixtures = pd.read_csv("C:\\Users\\tilly\\Downloads\\fixtures (3).csv")
filtered_file_path = 'C:\\Users\\tilly\\PycharmProjects\\GoalPrediction\\data_collection\\btts_all2.csv'
file_path = 'C:\\Users\\tilly\\PycharmProjects\\GoalPrediction\\data_collection\\o2.5results_data1.csv'

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the title of the dashboard
st.title('Football Team Statistics Dashboard')

# Display fixtures and let the user select one
selected_match = st.selectbox('Select a match', fixtures.apply(lambda x: f"{x['HomeTeam']} vs {x['AwayTeam']} - {x['Date']}", axis=1))

# Extract selected teams
home_team = fixtures[fixtures.apply(lambda x: f"{x['HomeTeam']} vs {x['AwayTeam']} - {x['Date']}", axis=1) == selected_match]['HomeTeam'].values[0]
away_team = fixtures[fixtures.apply(lambda x: f"{x['HomeTeam']} vs {x['AwayTeam']} - {x['Date']}", axis=1) == selected_match]['AwayTeam'].values[0]

# Create a selection box for leagues
leagues = data['Div'].unique()
selected_league = st.selectbox('Select a League:', leagues)

# Filter data for the selected league
league_data = data[data['Div'] == selected_league]

# Filter for matches on or after August 1, 2024
start_date = pd.Timestamp('2024-08-01')
league_data_filtered = league_data[league_data['Date'] >= start_date]

# Filter data for the last 5 matches for the selected home team
home_team_data_last_5 = league_data[league_data['HomeTeam'] == home_team].sort_values(by='Date').tail(5)

# Filter data for the last 5 matches for the selected away team
away_team_data_last_5 = league_data[league_data['AwayTeam'] == away_team].sort_values(by='Date').tail(5)

# Calculate average goals scored and conceded for the last 5 games
home_goals_scored_last_5 = home_team_data_last_5['FTHG'].mean()
home_goals_conceded_last_5 = home_team_data_last_5['FTAG'].mean()
away_goals_scored_last_5 = away_team_data_last_5['FTAG'].mean()
away_goals_conceded_last_5 = away_team_data_last_5['FTHG'].mean()

firsthalfhome_goals_scored_last_5 = home_team_data_last_5['HTHG'].mean()
firsthalfhome_goals_conceded_last_5 = home_team_data_last_5['HTAG'].mean()
firsthalfaway_goals_scored_last_5 = away_team_data_last_5['HTAG'].mean()
firsthalfaway_goals_conceded_last_5 = away_team_data_last_5['HTHG'].mean()

sechalfhome_goals_scored_last_5 = home_team_data_last_5['SHG'].mean()
sechalfhome_goals_conceded_last_5 = home_team_data_last_5['SAG'].mean()
sechalfaway_goals_scored_last_5 = away_team_data_last_5['SAG'].mean()
sechalfaway_goals_conceded_last_5 = away_team_data_last_5['SHG'].mean()

# Calculate average goals scored and conceded for both teams (using all matches)
home_goals_scored_overall = league_data[league_data['HomeTeam'] == home_team]['FTHG'].mean()
home_goals_conceded_overall = league_data[league_data['HomeTeam'] == home_team]['FTAG'].mean()
away_goals_scored_overall = league_data[league_data['AwayTeam'] == away_team]['FTAG'].mean()
away_goals_conceded_overall = league_data[league_data['AwayTeam'] == away_team]['FTHG'].mean()

firsthalfhome_goals_scored_overall = league_data[league_data['HomeTeam'] == home_team]['HTHG'].mean()
firsthalfhome_goals_conceded_overall = league_data[league_data['HomeTeam'] == home_team]['HTAG'].mean()
firsthalfaway_goals_scored_overall = league_data[league_data['AwayTeam'] == away_team]['HTAG'].mean()
firsthalfaway_goals_conceded_overall = league_data[league_data['AwayTeam'] == away_team]['HTHG'].mean()

sechalfhome_goals_scored_overall = league_data[league_data['HomeTeam'] == home_team]['SHG'].mean()
sechalfhome_goals_conceded_overall = league_data[league_data['HomeTeam'] == home_team]['SAG'].mean()
sechalfaway_goals_scored_overall = league_data[league_data['AwayTeam'] == away_team]['SAG'].mean()
sechalfaway_goals_conceded_overall = league_data[league_data['AwayTeam'] == away_team]['SHG'].mean()


# Display statistics for the last 5 games
st.subheader(f'📊 **Average Stats for Last 5 Games** - {home_team} (Home) and {away_team} (Away)')

# Display Home Team Stats
st.markdown(f"### 🏠 {home_team} (Home Stats)")
col1, col2, col3 = st.columns(3)
col1.metric(label="Avg Goals Scored", value=f"{home_goals_scored_last_5:.2f}")
col1.metric(label="Avg 1st Half Goals Scored", value=f"{firsthalfhome_goals_scored_last_5:.2f}")
col1.metric(label="Avg 2nd Half Goals Scored", value=f"{sechalfhome_goals_scored_last_5:.2f}")

col2.metric(label="Avg Goals Conceded", value=f"{home_goals_conceded_last_5:.2f}")
col2.metric(label="Avg 1st Half Goals Conceded", value=f"{firsthalfhome_goals_conceded_last_5:.2f}")
col2.metric(label="Avg 2nd Half Goals Conceded", value=f"{sechalfhome_goals_conceded_last_5:.2f}")

# Display Away Team Stats
st.markdown(f"### 🛫 {away_team} (Away Stats)")
col1, col2 = st.columns(2)
col1.metric(label="Avg Goals Scored", value=f"{away_goals_scored_last_5:.2f}")
col1.metric(label="Avg 1st Half Goals Scored", value=f"{firsthalfaway_goals_scored_last_5:.2f}")
col1.metric(label="Avg 2nd Half Goals Scored", value=f"{sechalfaway_goals_scored_last_5:.2f}")

col2.metric(label="Avg Goals Conceded", value=f"{away_goals_conceded_last_5:.2f}")
col2.metric(label="Avg 1st Half Goals Conceded", value=f"{firsthalfaway_goals_conceded_last_5:.2f}")
col2.metric(label="Avg 2nd Half Goals Conceded", value=f"{sechalfaway_goals_conceded_last_5:.2f}")

# Divider
st.markdown("---")

# Display overall statistics
st.subheader(f'🌐 **Overall Average Stats** - {home_team} (Home) and {away_team} (Away)')

# Home Overall Stats
st.markdown(f"### 🏠 {home_team} (Home Overall Stats)")
col1, col2 = st.columns(2)
col1.metric(label="Avg Goals Scored (Overall)", value=f"{home_goals_scored_overall:.2f}")
col1.metric(label="Avg 1st Half Goals Scored (Overall)", value=f"{firsthalfhome_goals_scored_overall:.2f}")
col1.metric(label="Avg 2nd Half Goals Scored (Overall)", value=f"{sechalfhome_goals_scored_overall:.2f}")

col2.metric(label="Avg Goals Conceded (Overall)", value=f"{home_goals_conceded_overall:.2f}")
col2.metric(label="Avg 1st Half Goals Conceded (Overall)", value=f"{firsthalfhome_goals_conceded_overall:.2f}")
col2.metric(label="Avg 2nd Half Goals Conceded (Overall)", value=f"{sechalfhome_goals_conceded_overall:.2f}")

# Away Overall Stats
st.markdown(f"### 🛫 {away_team} (Away Overall Stats)")
col1, col2 = st.columns(2)
col1.metric(label="Avg Goals Scored (Overall)", value=f"{away_goals_scored_overall:.2f}")
col1.metric(label="Avg 1st Half Goals Scored (Overall)", value=f"{firsthalfaway_goals_scored_overall:.2f}")
col1.metric(label="Avg 2nd Half Goals Scored (Overall)", value=f"{sechalfaway_goals_scored_overall:.2f}")

col2.metric(label="Avg Goals Conceded (Overall)", value=f"{away_goals_conceded_overall:.2f}")
col2.metric(label="Avg 1st Half Goals Conceded (Overall)", value=f"{firsthalfaway_goals_conceded_overall:.2f}")
col2.metric(label="Avg 2nd Half Goals Conceded (Overall)", value=f"{sechalfaway_goals_conceded_overall:.2f}")

# Divider for BTTS and O2.5 probabilities
st.markdown("---")
st.subheader("📊 **BTTS and O2.5 Probabilities**")

# Load BTTS data
# Load BTTS data
btts_data = pd.read_csv(filtered_file_path)

# Filter BTTS data for the selected teams
btts_prob_df = btts_data[(btts_data['HomeTeam'] == home_team) & (btts_data['AwayTeam'] == away_team)]

# Check if any data exists and extract the probability value
if not btts_prob_df.empty:
    btts_yes_prob = btts_prob_df['Probability'].values[0]  # Extract the first value of the 'Probability' column
    st.metric(label="BTTS Yes Probability", value=f"{btts_yes_prob:.2f}%")
else:
    st.warning("No BTTS data available for the selected match.")

# Load O2.5 data
o2_5_data = pd.read_csv(file_path)  # Use this if O2.5 data is in the same file

# Filter O2.5 data for the selected teams
o2_5_prob_df = o2_5_data[(o2_5_data['HomeTeam'] == home_team) & (o2_5_data['AwayTeam'] == away_team)]

# Check if any data exists and extract the probability value
if not o2_5_prob_df.empty:
    o2_5_yes_prob = o2_5_prob_df['Probability'].values[0]  # Extract the first value of the 'Probability' column
    st.metric(label="O 2.5 Yes Probability", value=f"{o2_5_yes_prob:.2f}%")
else:
    st.warning("No O 2.5 data available for the selected match.")




# Poisson Distribution Plot
st.subheader("Poisson Distribution for Predicted Goals")

# Simulate goal probabilities for both teams using Poisson
goal_range = range(0, 10)  # Creates a range from 0 to 9

home_poisson = [poisson.pmf(i, home_goals_scored_last_5) for i in goal_range]
away_poisson = [poisson.pmf(i, away_goals_scored_last_5) for i in goal_range]

# Plot Poisson Distribution for Home Team
fig, ax = plt.subplots()
ax.bar(goal_range, home_poisson, color='blue', alpha=0.7, label=f'{home_team} Goal Probability')
ax.set_title(f'{home_team} Goal Probability Distribution')
ax.set_xlabel('Goals')
ax.set_ylabel('Probability')
ax.legend()
st.pyplot(fig)

# Plot Poisson Distribution for Away Team
fig, ax = plt.subplots()
ax.bar(goal_range, away_poisson, color='red', alpha=0.7, label=f'{away_team} Goal Probability')
ax.set_title(f'{away_team} Goal Probability Distribution')
ax.set_xlabel('Goals')
ax.set_ylabel('Probability')
ax.legend()
st.pyplot(fig)

# Calculate probabilities for all scoreline combinations (home_goals, away_goals)
scoreline_probs = []
for home_goals in goal_range:
    for away_goals in goal_range:
        prob = poisson.pmf(home_goals, home_goals_scored_last_5) * poisson.pmf(away_goals, away_goals_scored_last_5)
        scoreline_probs.append((home_goals, away_goals, prob))

# Sort the scorelines by probability in descending order and pick the top 5
scoreline_probs = sorted(scoreline_probs, key=lambda x: x[2], reverse=True)[:5]

# Display the top 5 most probable scorelines
st.subheader(f'Top 5 Most Probable Scorelines for {home_team} vs {away_team}')
for home_goals, away_goals, prob in scoreline_probs:
    st.write(f'{home_team} {home_goals} - {away_team} {away_goals}: {prob:.2%}')

# Calculate Points Per Game (PPG) for home and away teams
home_games_played = len(league_data[league_data['HomeTeam'] == home_team])
away_games_played = len(league_data[league_data['AwayTeam'] == away_team])

home_points = (home_team_data_last_5['FTHG'] > home_team_data_last_5['FTAG']).sum() * 3 + \
              (home_team_data_last_5['FTHG'] == home_team_data_last_5['FTAG']).sum()
away_points = (away_team_data_last_5['FTAG'] > away_team_data_last_5['FTHG']).sum() * 3 + \
              (away_team_data_last_5['FTAG'] == away_team_data_last_5['FTHG']).sum()


home_ppg = home_points / home_games_played
away_ppg = away_points / away_games_played

# Adjust the Poisson probabilities based on PPG
ppg_factor = home_ppg / away_ppg if away_ppg != 0 else 1

# Calculate probabilities for all scoreline combinations (home_goals, away_goals)
scoreline_probs = []
home_win_prob = 0
away_win_prob = 0
draw_prob = 0

for home_goals in goal_range:
    for away_goals in goal_range:
        poisson_home_prob = poisson.pmf(home_goals, home_goals_scored_last_5)
        poisson_away_prob = poisson.pmf(away_goals, away_goals_scored_last_5)
        prob = poisson_home_prob * poisson_away_prob

        # Adjust based on PPG factor
        if home_goals > away_goals:
            home_win_prob += prob * ppg_factor
        elif away_goals > home_goals:
            away_win_prob += prob / ppg_factor
        else:
            draw_prob += prob

# Normalize the probabilities to ensure they sum to 100%
total_prob = home_win_prob + away_win_prob + draw_prob
home_win_prob /= total_prob
away_win_prob /= total_prob
draw_prob /= total_prob

# Display the probabilities for each outcome
st.subheader(f'Adjusted Probabilities for {home_team} vs {away_team}')
st.write(f"🏠 Home Win Probability (Adjusted): {home_win_prob:.2%}")
st.write(f"🛫 Away Win Probability (Adjusted): {away_win_prob:.2%}")
st.write(f"🤝 Draw Probability (Adjusted): {draw_prob:.2%}")



# Head-to-Head Comparison for the entire dataset
head_to_head = league_data[((league_data['HomeTeam'] == home_team) & (league_data['AwayTeam'] == away_team)) |
                            ((league_data['HomeTeam'] == away_team) & (league_data['AwayTeam'] == home_team))]

st.subheader('Head-to-Head Statistics (Entire Dataset)')
st.write(f"Matches Played: {len(head_to_head)}")

# Calculate and display H2H wins, draws, and total goals
home_wins = len(head_to_head[(head_to_head['HomeTeam'] == home_team) & (head_to_head['FTHG'] > head_to_head['FTAG'])])
away_wins = len(head_to_head[(head_to_head['AwayTeam'] == away_team) & (head_to_head['FTAG'] > head_to_head['FTHG'])])
draws = len(head_to_head[head_to_head['FTHG'] == head_to_head['FTAG']])
total_goals_home = head_to_head[head_to_head['HomeTeam'] == home_team]['FTHG'].sum()
total_goals_away = head_to_head[head_to_head['AwayTeam'] == away_team]['FTAG'].sum()

st.write(f"Total Goals Scored (Home): {total_goals_home}")
st.write(f"Total Goals Scored (Away): {total_goals_away}")
st.write(f"Home Wins: {home_wins}")
st.write(f"Away Wins: {away_wins}")
st.write(f"Draws: {draws}")

# Displaying each H2H result
st.subheader('Head-to-Head Results')
if not head_to_head.empty:
    h2h_results = head_to_head[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'SHG', 'SAG']]
    h2h_results['Result'] = h2h_results.apply(lambda row: 'Home Win' if row['FTHG'] > row['FTAG'] else ('Away Win' if row['FTHG'] < row['FTAG'] else 'Draw'), axis=1)
    st.write(h2h_results)
else:
    st.write("No matches found between the selected teams.")


# Filter data from August 1st, 2024
season_data = data[data['Date'] >= '2024-08-01']

# Streamlit App
st.title("Football League Table")

# Select division
division = st.selectbox("Select Division", season_data['Div'].unique())

# Filter data for the selected division
league_data_filtered = season_data[season_data['Div'] == division]

# Calculate the league table for home teams
home_league_table = league_data_filtered.groupby('HomeTeam').agg(
    Wins=('FTHG', lambda x: (x > league_data_filtered.loc[x.index, 'FTAG']).sum()),
    Losses=('FTHG', lambda x: (x < league_data_filtered.loc[x.index, 'FTAG']).sum()),
    Draws=('FTHG', lambda x: (x == league_data_filtered.loc[x.index, 'FTAG']).sum()),
    Goals_Scored=('FTHG', 'sum'),
    Goals_Conceded=('FTAG', 'sum')
).reset_index()

# Calculate points and goal difference for home table
home_league_table['Points'] = home_league_table['Wins'] * 3 + home_league_table['Draws']
home_league_table['Goal_Difference'] = home_league_table['Goals_Scored'] - home_league_table['Goals_Conceded']

# Sort the home table by Points and Goal Difference
home_league_table = home_league_table.sort_values(by=['Points', 'Goal_Difference'], ascending=False).reset_index(drop=True)

# Calculate the league table for away teams
away_league_table = league_data_filtered.groupby('AwayTeam').agg(
    Wins=('FTAG', lambda x: (x > league_data_filtered.loc[x.index, 'FTHG']).sum()),
    Losses=('FTAG', lambda x: (x < league_data_filtered.loc[x.index, 'FTHG']).sum()),
    Draws=('FTAG', lambda x: (x == league_data_filtered.loc[x.index, 'FTHG']).sum()),
    Goals_Scored=('FTAG', 'sum'),
    Goals_Conceded=('FTHG', 'sum')
).reset_index()

# Rename columns in away_league_table for clarity
away_league_table.rename(columns={'AwayTeam': 'HomeTeam'}, inplace=True)

# Calculate points and goal difference for away table
away_league_table['Points'] = away_league_table['Wins'] * 3 + away_league_table['Draws']
away_league_table['Goal_Difference'] = away_league_table['Goals_Scored'] - away_league_table['Goals_Conceded']

# Sort the away table by Points and Goal Difference
away_league_table = away_league_table.sort_values(by=['Points', 'Goal_Difference'], ascending=False).reset_index(drop=True)

# Merge home and away tables
league_table = pd.merge(home_league_table, away_league_table, how='outer', on='HomeTeam', suffixes=('_Home', '_Away'))

# Combine results
league_table['Wins'] = league_table['Wins_Home'].fillna(0) + league_table['Wins_Away'].fillna(0)
league_table['Losses'] = league_table['Losses_Home'].fillna(0) + league_table['Losses_Away'].fillna(0)
league_table['Draws'] = league_table['Draws_Home'].fillna(0) + league_table['Draws_Away'].fillna(0)
league_table['Goals_Scored'] = league_table['Goals_Scored_Home'].fillna(0) + league_table['Goals_Scored_Away'].fillna(0)
league_table['Goals_Conceded'] = league_table['Goals_Conceded_Home'].fillna(0) + league_table['Goals_Conceded_Away'].fillna(0)

# Keep relevant columns and calculate points
league_table = league_table[['HomeTeam', 'Wins', 'Losses', 'Draws', 'Goals_Scored', 'Goals_Conceded']]
league_table['Points'] = league_table['Wins'] * 3 + league_table['Draws']

# Sort the combined table by Points and Goal Difference
league_table['Goal_Difference'] = league_table['Goals_Scored'] - league_table['Goals_Conceded']
league_table = league_table.sort_values(by=['Points', 'Goal_Difference'], ascending=False).reset_index(drop=True)

# Display the combined table by default
st.write("**Combined League Table**")
st.write(league_table)

# Create tabs for home and away tables
tab1, tab2 = st.tabs(["Home Table", "Away Table"])

with tab1:
    st.write("**Home League Table**")
    st.write(home_league_table)

with tab2:
    st.write("**Away League Table**")
    st.write(away_league_table)



# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.sort_values('Date')  # Sort data by date for calculating recent matches

# Function to calculate team metrics
def calculate_team_metrics(data, last_n_games=None):
    teams = pd.concat([data['HomeTeam'], data['AwayTeam']]).unique()
    metrics = pd.DataFrame(index=teams)

    # Initialize columns for metrics
    metrics['Avg Goals Scored 1H (Overall)'] = 0
    metrics['Avg Goals Conceded 1H (Overall)'] = 0
    metrics['Avg Goals Scored 1H (Last 5)'] = 0
    metrics['Avg Goals Conceded 1H (Last 5)'] = 0
    metrics['Games Played'] = 0

    # Calculate metrics for each team
    for team in teams:
        home_matches = data[data['HomeTeam'] == team]
        away_matches = data[data['AwayTeam'] == team]

        # For overall metrics
        total_home_games = len(home_matches)
        total_away_games = len(away_matches)

        goals_scored_1H_home = home_matches['HTHG'].sum()  # Home team's first half goals
        goals_scored_1H_away = away_matches['HTAG'].sum()  # Away team's first half goals
        goals_conceded_1H_home = home_matches['HTAG'].sum()  # Home conceded goals (away team)
        goals_conceded_1H_away = away_matches['HTHG'].sum()  # Away conceded goals (home team)

        total_goals_scored_1H = goals_scored_1H_home + goals_scored_1H_away
        total_goals_conceded_1H = goals_conceded_1H_home + goals_conceded_1H_away
        games_played = total_home_games + total_away_games

        # For recent form (Last 5 games)
        recent_home_matches = home_matches.tail(last_n_games) if last_n_games else home_matches
        recent_away_matches = away_matches.tail(last_n_games) if last_n_games else away_matches

        recent_goals_scored_1H_home = recent_home_matches['HTHG'].sum()  # Last N home games 1st half goals
        recent_goals_scored_1H_away = recent_away_matches['HTAG'].sum()  # Last N away games 1st half goals
        recent_goals_conceded_1H_home = recent_home_matches['HTAG'].sum()  # Last N home conceded 1st half goals
        recent_goals_conceded_1H_away = recent_away_matches['HTHG'].sum()  # Last N away conceded 1st half goals

        total_recent_goals_scored_1H = recent_goals_scored_1H_home + recent_goals_scored_1H_away
        total_recent_goals_conceded_1H = recent_goals_conceded_1H_home + recent_goals_conceded_1H_away
        recent_games_played = len(recent_home_matches) + len(recent_away_matches)

        # Update metrics
        metrics.at[team, 'Avg Goals Scored 1H (Overall)'] = total_goals_scored_1H / games_played if games_played > 0 else 0
        metrics.at[team, 'Avg Goals Conceded 1H (Overall)'] = total_goals_conceded_1H / games_played if games_played > 0 else 0
        metrics.at[team, 'Avg Goals Scored 1H (Last 5)'] = total_recent_goals_scored_1H / recent_games_played if recent_games_played > 0 else 0
        metrics.at[team, 'Avg Goals Conceded 1H (Last 5)'] = total_recent_goals_conceded_1H / recent_games_played if recent_games_played > 0 else 0
        metrics.at[team, 'Games Played'] = games_played

    return metrics

# Calculate metrics for both overall and last 5 games
metrics = calculate_team_metrics(data, last_n_games=5)


# Filter dataset for a specific season (e.g., 2024 season)
season_data = data[(data['Date'] >= '2024-08-01')]

# Initialize Elo ratings with a fixed number of points
initial_points = 1000
elo_ratings = {team: initial_points for team in pd.concat([season_data['HomeTeam'], season_data['AwayTeam']]).unique()}

# Function to calculate expected result
def expected_result(home_elo, away_elo, home_field_advantage=0):
    return 1 / (1 + 10 ** ((away_elo - (home_elo + home_field_advantage)) / 400))

# Function to calculate goal factor based on goal difference
def goal_factor(goal_diff):
    return np.log(goal_diff + 1)

# Elo rating calculation
def calculate_elo(data, k=32, home_field_advantage=50):
    global elo_ratings
    results = []

    for index, row in data.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_score = row['FTHG']
        away_score = row['FTAG']

        # Get current Elo ratings
        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        # Calculate goal difference
        goal_diff = abs(home_score - away_score)
        g_factor = goal_factor(goal_diff)

        # Calculate the expected result (including home-field advantage)
        expected_home = expected_result(home_elo, away_elo, home_field_advantage=home_field_advantage)

        if home_score > away_score:  # Home Win
            actual_result = 1  # Home win
        elif away_score > home_score:  # Away Win
            actual_result = 0  # Away win
        else:
            actual_result = 0.5  # Draw

        # Update Elo ratings
        elo_ratings[home_team] += k * g_factor * (actual_result - expected_home)
        elo_ratings[away_team] += k * g_factor * (expected_home - actual_result)

        results.append({
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'ActualResult': 'Home Win' if home_score > away_score else 'Away Win' if away_score > home_score else 'Draw',
            'HomeElo': elo_ratings[home_team],
            'AwayElo': elo_ratings[away_team],
            'GoalDiff': goal_diff
        })

    return pd.DataFrame(results)

# Calculate Elo ratings from the data
calculate_elo(season_data)

# Streamlit UI
st.title("Football Elo Rating System")

# Select two teams from the available teams
teams = list(elo_ratings.keys())
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

# Display current Elo ratings for the selected teams
st.write(f"**{home_team} Elo Rating:** {elo_ratings[home_team]:.2f}")
st.write(f"**{away_team} Elo Rating:** {elo_ratings[away_team]:.2f}")

# You could also add a button to trigger an update of ratings
if st.button("Update Ratings"):
    st.warning("Please implement functionality to simulate or input match results.")
# Optional: Display all Elo ratings if needed
if st.checkbox("Show All Elo Ratings"):
    st.write(pd.DataFrame(elo_ratings.items(), columns=['Team', 'Elo Rating']).sort_values(by='Elo Rating', ascending=False))
