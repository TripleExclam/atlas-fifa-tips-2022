import os

import json
import pandas as pd
import statsmodels.api as sm

from flask import Flask, make_response, jsonify, request
from functools import wraps
from jsonschema import validate, ValidationError

app = Flask(__name__)

schema = {
    "type" : "object",
    "properties" : {
        "round": {"type" : "string"},
        "match": {"type" : "number"},
        "team1": {"type" : "string"},
        "team2": {"type" : "string"},
        "results": {"type": "array"}
    },
    "required": ["round", "match", "team1", "team2"]
}

def set_response_security_headers(api_func):
    @wraps(api_func)
    def wrapper(*args, **kwargs):
        response = api_func(*args, **kwargs)
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'none'; form-action 'none'; frame-ancestors 'none'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response

    return wrapper

def group_stage_pred(team1, team2):
    df = pd.read_csv('tips.csv')

    winner = df.loc[
        ((df.HomeTeam == team1) & (df.AwayTeam == team2)) |
        ((df.HomeTeam == team2) & (df.AwayTeam == team1)), 
        'winning_team'
    ]
    if len(winner) == 0:
        return None

    return winner.iloc[0]

def adjust_eg(goals, elo_diff):
    model_slope = 0.00174229
    expected_goals = lambda x: x * model_slope
    elo_diff_multipliter = expected_goals(elo_diff)
    return goals + elo_diff_multipliter

def knockout_pred(team1, team2, results):
    K = 50
    df = pd.read_csv('model_features.csv')
    model = sm.load('elo_model.pickle')
    
    if team1 not in df.team.unique() or team2 not in df.team.unique():
        return team1

    for r in results:
        try:
            t1 = r['team1']['name']
            t2 = r['team2']['name']
            s1 = r['team1']['score']
            s2 = r['team2']['score']

            e1 = df[df.team == t1].elo.iloc[0]
            e2 = df[df.team == t2].elo.iloc[0]
            W1 = 1 if (s1 > s2) else 0.5 if (s1 == s2) else 0
            W2 = 1 - W1
            W_e1 = 1 / (10 ** (-(e1 - e2) / 400) + 1)
            W_e2 = 1 / (10 ** (-(e2 - e1) / 400) + 1)

            a1 = e1 + K * (W1 - W_e1)
            a2 = e2 + K * (W2 - W_e2)
            print("elo changes", e1, "to", a1, e2, "to", a2)
            df.loc[df.team == t1, 'elo'] = a1
            df.loc[df.team == t2, 'elo'] = a2
        except:
            print("failed to parse result", r)

    elo1 = df[df.team == team1].elo.iloc[0]
    elo2 = df[df.team == team2].elo.iloc[0]
    xg1 = adjust_eg(df[df.team == team1].mean_goals.iloc[0], elo1 - elo2)
    xg2 = adjust_eg(df[df.team == team2].mean_goals.iloc[0], elo2 - elo1)

    feature_vec = [xg1, xg2, elo1, elo2]

    result = model.predict(feature_vec)

    return team1 if result > 0.5 else team2


@app.route('/tip', methods=['POST'])
@set_response_security_headers
def tip_request():
    data = request.json

    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        message = f"Invalid schema provided {e.message}"
        return make_response(jsonify(message=message), 400)

    round = data['round']
    team1 = data['team1']
    team2 = data['team2']
    results = data.get('results', [])

    if round == 'Group':
        winner = group_stage_pred(team1, team2)
    else:
        winner = knockout_pred(team1, team2, results)

    if winner is None:
        return make_response(jsonify(message='Invalid matchup.'), 400)
    
    return make_response(jsonify(winner=winner), 200)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))