import pickle

import xgboost as xgb
from flask import Flask, jsonify, request

MODEL_PATH = 'model.bin'
THRESHOLD = 0.35

AUDIO = ['bpm', 'danceability', 'valence', 'energy', 'acousticness',
         'instrumentalness', 'liveness', 'speechiness']
FEATURES = AUDIO + ['key', 'mode', 'artist_count', 'released_month', 'released_year']
CATEGORICAL = ['key', 'mode', 'released_month']

with open(MODEL_PATH, 'rb') as f_in:
    dv, model = pickle.load(f_in)

FEATURE_NAMES = list(dv.get_feature_names_out())
ALLOWED = {
    c: sorted({n.split('=', 1)[1] for n in FEATURE_NAMES if n.startswith(c + '=')},
              key=lambda v: (len(v), v))
    for c in CATEGORICAL
}

app = Flask('mega-hit')


def prepare(song):
    missing = [f for f in FEATURES if f not in song]
    if missing:
        raise ValueError(f'missing fields: {missing}')

    row = {}
    for f in FEATURES:
        if f in CATEGORICAL:
            value = str(song[f])
            if value not in ALLOWED[f]:
                raise ValueError(f'{f} must be one of {ALLOWED[f]}')
            row[f] = value
        else:
            row[f] = float(song[f])
    return row


def predict_single(row, dv, model):
    X = dv.transform([row])
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    return float(model.predict(dmatrix)[0])


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    song = request.get_json(silent=True)
    if not isinstance(song, dict):
        return jsonify({'error': 'send a JSON object with the song features'}), 400

    try:
        row = prepare(song)
    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400

    probability = predict_single(row, dv, model)

    return jsonify({
        'mega_hit_probability': probability,
        'mega_hit': bool(probability >= THRESHOLD),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)