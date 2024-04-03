from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)
CORS(app)

# Load the model
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.get_json()
        features = data['features']
        prediction = model.predict_proba([features])
        output = {'predictions': prediction.tolist()[0]}
        return jsonify(output)
    except Exception as e:
        print(e)
        return jsonify(error='Something went wrong'), 500

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error='The requested URL was not found on the server'), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
