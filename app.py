from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load and preprocess data
file_path = "C:\\Users\\tilly\\OneDrive\\Desktop\\FootballData\\updated_data2.csv"
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])

def get_teams_by_league(league):
    if league:
        teams = data[data['Div'] == league]['HomeTeam'].unique().tolist()
        teams = [str(team) for team in teams if pd.notna(team) and team is not None]
        return teams
    return []

def calculate_stats(league, home_team, away_team, num_games, selected_stat):
    filtered_data = data[data['Div'] == league]

    home_results = []
    away_results = []

    if home_team:
        team_data_home = filtered_data[filtered_data['HomeTeam'] == home_team]
        last_games_home = team_data_home.tail(num_games)
        if not last_games_home.empty:
            if selected_stat == 'Goals':
                avg_goals_home = last_games_home['FTHG'].mean()
                avg_goals_against_home = last_games_home['FTAG'].mean()
                avg_sot_home = last_games_home['HST'].mean()
                g_per_sot_home = avg_goals_home / avg_sot_home if avg_sot_home > 0 else 0
                avg_sot_home = last_games_home['HST'].mean()
                avg_sotagainst_home = last_games_home['AST'].mean()
                home_results.append(f"Avg Shots on Target (Home - {home_team}): {avg_sot_home:.2f}")
                home_results.append(f"Avg Shots on Target Against(Home - {home_team}): {avg_sotagainst_home:.2f}")
                home_results.append(f"Avg Goals (Home - {home_team}): {avg_goals_home:.2f}")
                home_results.append(f"Avg Goals Against (Home - {home_team}): {avg_goals_against_home:.2f}")
                home_results.append(f"Goals per SoT (Home - {home_team}): {g_per_sot_home:.2f}")

            if selected_stat == 'Over 0.5 Goals':
                over_0_5_goals_home = (last_games_home['FTHG'] + last_games_home['FTAG']) > 0.5
                prob_over_0_5_home = over_0_5_goals_home.mean()
                home_results.append(f"Probability of Over 0.5 Goals (Home - {home_team}): {prob_over_0_5_home:.2%}")

            if selected_stat == 'Over 1.5 Goals':
                over_1_5_goals_home = (last_games_home['FTHG'] + last_games_home['FTAG']) > 1.5
                prob_over_1_5_home = over_1_5_goals_home.mean()
                home_results.append(f"Probability of Over 1.5 Goals (Home - {home_team}): {prob_over_1_5_home:.2%}")

            if selected_stat == 'Over 2.5 Goals':
                over_2_5_goals_home = (last_games_home['FTHG'] + last_games_home['FTAG']) > 2.5
                prob_over_2_5_home = over_2_5_goals_home.mean()
                home_results.append(f"Probability of Over 2.5 Goals (Home - {home_team}): {prob_over_2_5_home:.2%}")

            if selected_stat == 'Over 3.5 Goals':
                over_3_5_goals_home = (last_games_home['FTHG'] + last_games_home['FTAG']) > 3.5
                prob_over_3_5_home = over_3_5_goals_home.mean()
                home_results.append(f"Probability of Over 3.5 Goals (Home - {home_team}): {prob_over_3_5_home:.2%}")

            if selected_stat == 'Over 4.5 Goals':
                over_4_5_goals_home = (last_games_home['FTHG'] + last_games_home['FTAG']) > 4.5
                prob_over_4_5_home = over_4_5_goals_home.mean()
                home_results.append(f"Probability of Over 4.5 Goals (Home - {home_team}): {prob_over_4_5_home:.2%}")

            if selected_stat == 'Corners':
                avg_corners_home = last_games_home['HC'].mean()
                home_results.append(f"Avg Corners (Home - {home_team}): {avg_corners_home:.2f}")
                avg_cornersagainst_home = last_games_home['AC'].mean()
                home_results.append(f"Avg Corners Against (Home - {home_team}): {avg_cornersagainst_home:.2f}")

            if selected_stat == 'YellowCards':
                avg_yellow_home = last_games_home['HY'].mean()
                home_results.append(f"Avg Yellow Cards (Home - {home_team}): {avg_yellow_home:.2f}")
                avg_yellowcardsagainst_home = last_games_home['AY'].mean()
                home_results.append(f"Avg Yellow Cards Against (Home - {home_team}): {avg_yellowcardsagainst_home:.2f}")

            if selected_stat == 'RedCards':
                avg_red_home = last_games_home['HR'].mean()
                home_results.append(f"Avg Red Cards (Home - {home_team}): {avg_red_home:.2f}")
                avg_redcardsagainst_home = last_games_home['AR'].mean()
                home_results.append(f"Avg Red Cards Against (Home - {home_team}): {avg_redcardsagainst_home:.2f}")

            if selected_stat == 'Shots on Target':
                avg_sot_home = last_games_home['HST'].mean()
                home_results.append(f"Avg Shots on Target (Home - {home_team}): {avg_sot_home:.2f}")

            if selected_stat == 'Last Results':
                home_lastresult = last_games_home['FTR'].tolist()
                results_str = ", ".join(home_lastresult)
                home_results.append(f"Results (Home - {home_team}): {results_str}")



        else:
            home_results.append(f"No data available for home games of {home_team}.")

    if away_team:
        team_data_away = filtered_data[filtered_data['AwayTeam'] == away_team]
        last_games_away = team_data_away.tail(num_games)
        if not last_games_away.empty:
            if selected_stat == 'Goals':
                avg_goals_away = last_games_away['FTAG'].mean()
                avg_goals_against_away = last_games_away['FTHG'].mean()
                avg_sot_away = last_games_away['AST'].mean()
                g_per_sot_away = avg_goals_away / avg_sot_away if avg_sot_away > 0 else 0
                avg_sot_away = last_games_away['AST'].mean()
                avg_sotagainst_away = last_games_away['HST'].mean()
                away_results.append(f"Avg Shots on Target (Away - {away_team}): {avg_sot_away:.2f}")
                away_results.append(f"Avg Shots on Target Against(Away - {away_team}): {avg_sotagainst_away:.2f}")
                away_results.append(f"Avg Goals (Away - {away_team}): {avg_goals_away:.2f}")
                away_results.append(f"Avg Goals Against (Away - {away_team}): {avg_goals_against_away:.2f}")
                away_results.append(f"Goals per SoT (Away - {away_team}): {g_per_sot_away:.2f}")

            if selected_stat == 'Over 0.5 Goals':
                over_0_5_goals_away = (last_games_away['FTHG'] + last_games_away['FTAG']) > 0.5
                prob_over_0_5_away = over_0_5_goals_away.mean()
                away_results.append(f"Probability of Over 0.5 Goals (Away - {away_team}): {prob_over_0_5_away:.2%}")

            if selected_stat == 'Over 1.5 Goals':
                over_1_5_goals_away = (last_games_away['FTHG'] + last_games_away['FTAG']) > 1.5
                prob_over_1_5_away = over_1_5_goals_away.mean()
                away_results.append(f"Probability of Over 1.5 Goals (Away - {away_team}): {prob_over_1_5_away:.2%}")

            if selected_stat == 'Over 2.5 Goals':
                over_2_5_goals_away = (last_games_away['FTHG'] + last_games_away['FTAG']) > 2.5
                prob_over_2_5_away = over_2_5_goals_away.mean()
                away_results.append(f"Probability of Over 2.5 Goals (Away - {away_team}): {prob_over_2_5_away:.2%}")

            if selected_stat == 'Over 3.5 Goals':
                over_3_5_goals_away = (last_games_away['FTHG'] + last_games_away['FTAG']) > 3.5
                prob_over_3_5_away = over_3_5_goals_away.mean()
                away_results.append(f"Probability of Over 3.5 Goals (Away - {away_team}): {prob_over_3_5_away:.2%}")

            if selected_stat == 'Over 4.5 Goals':
                over_4_5_goals_away = (last_games_away['FTHG'] + last_games_away['FTAG']) > 4.5
                prob_over_4_5_away = over_4_5_goals_away.mean()
                away_results.append(f"Probability of Over 4.5 Goals (Away - {away_team}): {prob_over_4_5_away:.2%}")

            if selected_stat == 'Corners':
                avg_corners_away = last_games_away['AC'].mean()
                away_results.append(f"Avg Corners (Away - {away_team}): {avg_corners_away:.2f}")
                avg_cornersagainst_away = last_games_away['HC'].mean()
                away_results.append(f"Avg Corners Against (Away - {away_team}): {avg_cornersagainst_away:.2f}")

            if selected_stat == 'YellowCards':
                avg_yellow_away = last_games_away['AY'].mean()
                away_results.append(f"Avg Yellow Cards (Away - {away_team}): {avg_yellow_away:.2f}")
                avg_yellowcardsagainst_away = last_games_away['HY'].mean()
                away_results.append(f"Avg Yellow Cards Against (Away - {away_team}): {avg_yellowcardsagainst_away:.2f}")

            if selected_stat == 'RedCards':
                avg_red_away = last_games_away['AR'].mean()
                away_results.append(f"Avg Red Cards (Away - {away_team}): {avg_red_away:.2f}")
                avg_redcardsagainst_away = last_games_away['HR'].mean()
                away_results.append(f"Avg Red Cards Against (Away - {away_team}): {avg_redcardsagainst_away:.2f}")

            if selected_stat == 'Shots on Target':
                avg_sot_away = last_games_away['AST'].mean()
                away_results.append(f"Avg Shots on Target (Away - {away_team}): {avg_sot_away:.2f}")

            if selected_stat == 'Last Results':
                away_lastresult = last_games_away['FTR'].tolist()
                results_str = ", ".join(away_lastresult)
                away_results.append(f"Results (Away - {away_team}): {results_str}")





        else:
            away_results.append(f"No data available for away games of {away_team}.")

    return home_results, away_results

@app.route('/')
def index():
    leagues = data['Div'].unique().tolist()
    stats = [
        'Goals', 'Corners', 'YellowCards', 'Shots on Target', 'RedCards',
        'Over 0.5 Goals', 'Over 1.5 Goals', 'Over 2.5 Goals', 'Over 3.5 Goals', 'Over 4.5 Goals', 'Last Results',
        'Points Per Game'
    ]
    return render_template('index.html', leagues=leagues, stats=stats)

@app.route('/teams', methods=['GET'])
def teams():
    league = request.args.get('league')
    teams_list = get_teams_by_league(league)
    return jsonify(teams_list)

@app.route('/calculate', methods=['POST'])
def calculate():
    league = request.form.get('league')
    home_team = request.form.get('home_team')
    away_team = request.form.get('away_team')
    num_games = int(request.form.get('num_games'))
    selected_stat = request.form.get('stat')

    home_results, away_results = calculate_stats(league, home_team, away_team, num_games, selected_stat)

    return render_template('results.html', home_results=home_results, away_results=away_results)

if __name__ == '__main__':
    app.run(debug=True)
