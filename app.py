from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle 

app = Flask(__name__)

ipl=pd.read_csv("ipl_data.csv")
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis =1)
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

venues = df['venue'].unique().tolist()
batting_teams = df['bat_team'].unique().tolist()
bowling_teams = df['bowl_team'].unique().tolist()
strikers = df['batsman'].unique().tolist()
bowlers = df['bowler'].unique().tolist()

df['venue'] = venue_encoder.fit_transform(df['venue'])
df['bat_team'] = batting_team_encoder.fit_transform(df['bat_team'])
df['bowl_team'] = bowling_team_encoder.fit_transform(df['bowl_team'])
df['batsman'] = striker_encoder.fit_transform(df['batsman'])
df['bowler'] = bowler_encoder.fit_transform(df['bowler'])

scaler = MinMaxScaler()
scaler.fit(df[['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']])


model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def index():
    return render_template('index.html', venues=venues, batting_team=batting_teams,bowling_team=bowling_teams, striker=strikers,bowler=bowlers)

@app.route('/predict', methods=['POST'])
def predict():
    venue = request.form['venue']
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    striker = request.form['batsman']
    bowler = request.form['bowler']
    
    decoded_venue = venue_encoder.transform([venue])
    decoded_batting_team = batting_team_encoder.transform([batting_team])
    decoded_bowling_team = bowling_team_encoder.transform([bowling_team])
    decoded_striker = striker_encoder.transform([striker])
    decoded_bowler = bowler_encoder.transform([bowler])


    input = np.array([decoded_venue,  decoded_batting_team, decoded_bowling_team,decoded_striker, decoded_bowler])
    input = input.reshape(1,5)
    input = scaler.transform(input)
    #print(input)
    predicted_score = model.predict(input)
    predicted_score = int(predicted_score[0,0])

    # print(predicted_score)
    
    return render_template('index.html', prediction_text=f"Prediction: Total runs {predicted_score} can be achieved by team {batting_team} ", venues=venues, batting_team=batting_teams, bowling_team=bowling_teams, striker=strikers, bowler=bowlers)
    # return render_template('index.html', prediction_text=f"Predicted Score: {predicted_score}")
                           
if __name__ == "__main__":
    app.run(debug=True)
