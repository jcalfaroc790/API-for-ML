from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump, load
import os
import pandas as pd

# Directory to store trained models
MODEL_DIR = r'C:\Users\JUAN\Desktop\III2024\ML&BD\models'
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)

# Global variable to store loaded data
loaded_data = None

# Available models with their respective classes
MODEL_CLASSES = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "AdaBoost": AdaBoostClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC
}

# Endpoint to list available model classes for training
@app.route('/models', methods=['GET'])
def list_models():
    """
    Endpoint to return a list of available model classes for training.
    """
    return jsonify(list(MODEL_CLASSES.keys()))

# Endpoint to upload data via CSV
@app.route('/upload_data', methods=['POST'])
def upload_data():
    """
    Endpoint to upload data as a CSV file.
    """
    global loaded_data
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    try:
        loaded_data = pd.read_csv(file)
        num_records = len(loaded_data)
        return jsonify({"message": f"Data loaded successfully with {num_records} records"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

# Endpoint to train a model with optional hyperparameter tuning
@app.route('/train', methods=['POST'])
def train_model():
    global loaded_data
    data = request.get_json()
    model_name = data.get('model_name')
    params = data.get('params', {})

    if model_name not in MODEL_CLASSES:
        return jsonify({"error": "Model not available"}), 400

    if loaded_data is None:
        return jsonify({"error": "No data loaded. Please upload data first."}), 400

    # Clean data (remove rows with missing values)
    if loaded_data.isnull().values.any():
        loaded_data = loaded_data.dropna()

    if 'target' not in loaded_data.columns:
        return jsonify({"error": "Target column is missing in the data"}), 400

    # Prepare the dataset
    X = loaded_data.drop(columns=['target']).values  # Drop 'target' column for features
    y = loaded_data['target']  # Ensure 'target' is present as the target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model instantiation and hyperparameter tuning (use RandomizedSearchCV or GridSearchCV)
    model_class = MODEL_CLASSES[model_name]
    model = model_class()

    # Example: Randomized search with a smaller parameter space
    param_dist = {
        'n_estimators': [10, 50, 100],
        'max_depth': [5, 10, 15, None]
    }
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    dump(best_model, model_path)

    return jsonify({
        "message": f"{model_name} trained successfully",
        "best_params": random_search.best_params_
    })

# Endpoint to make predictions with a trained model
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions using a trained model.
    """
    data = request.get_json()
    model_name = data.get('model_name')
    input_data = data.get('input_data')

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404

    model = load(model_path)
    prediction = model.predict([input_data])
    return jsonify({"prediction": prediction.tolist()})

# Endpoint to delete a trained model
@app.route('/delete_model', methods=['DELETE'])
def delete_model():
    """
    Endpoint to delete a previously trained model.
    """
    data = request.get_json()
    model_name = data.get('model_name')

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(model_path):
        os.remove(model_path)
        return jsonify({"message": f"{model_name} deleted successfully"})
    else:
        return jsonify({"error": "Model not found"}), 404

# Endpoint to retrain a model (uses train_model function)
@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Endpoint to retrain an already existing model. 
    If the model doesn't exist, an error message will be returned.
    """
    data = request.get_json()
    model_name = data.get('model_name')
    params = data.get('params', {})

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        return jsonify({"error": f"{model_name} not found. Please train the model first."}), 404

    # Check if data is available and retrain
    if loaded_data is None:
        return jsonify({"error": "No data loaded. Please upload data first."}), 400

    X = loaded_data.drop(columns=['target']).values  # Drop 'target' column for features
    y = loaded_data['target']  # Ensure 'target' is present as the target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_class = MODEL_CLASSES[model_name]
    model = model_class()
    grid_search = GridSearchCV(model, params, cv=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Save the retrained model
    dump(best_model, model_path)
    
    return jsonify({
        "message": f"{model_name} retrained successfully",
        "best_params": grid_search.best_params_
    })

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint to check if the API is running and healthy.
    """
    return jsonify({"status": "API is running"})

if __name__ == '__main__':
    app.run(debug=True)
