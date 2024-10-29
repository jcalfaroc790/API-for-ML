from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from joblib import dump, load
import os

# Directory to store trained models
MODEL_DIR = r'C:\Users\JUAN\Desktop\III2024\ML&BD\models'
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)

# Available models
MODEL_CLASSES = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier
}

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify(list(MODEL_CLASSES.keys()))

@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    model_name = data.get('model_name')
    params = data.get('params', {})

    if model_name not in MODEL_CLASSES:
        return jsonify({"error": "Model not available"}), 400

    X, y = make_classification(n_samples=1000, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_class = MODEL_CLASSES[model_name]
    model = model_class()

    grid_search = GridSearchCV(model, params, cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    dump(best_model, model_path)

    return jsonify({"message": f"{model_name} trained successfully", "best_params": grid_search.best_params_})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data.get('model_name')
    input_data = data.get('input_data')

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404

    model = load(model_path)
    prediction = model.predict([input_data])

    return jsonify({"prediction": prediction.tolist()})

@app.route('/delete_model', methods=['DELETE'])
def delete_model():
    data = request.get_json()
    model_name = data.get('model_name')

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(model_path):
        os.remove(model_path)
        return jsonify({"message": f"{model_name} deleted successfully"})
    else:
        return jsonify({"error": "Model not found"}), 404

@app.route('/retrain', methods=['POST'])
def retrain_model():
    return train_model()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"})

if __name__ == '__main__':
    app.run(debug=True)
