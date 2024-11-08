# Machine Learning API for Model Training and Predictions

This repository contains a Python-based API designed for training machine learning models, hyperparameter tuning, and making predictions. The API allows users to upload datasets, train models, perform hyperparameter tuning, make predictions, and manage multiple models, including the ability to retrain and delete models.

## Features

- **Model Training**: Supports training multiple models with hyperparameter tuning.
- **Model Management**: Retrain or delete models when necessary.
- **Predictions**: Make predictions using trained models.
- **Health Check**: Check the status of the API.
- **Telegram Bot** (Bonus): Interact with the API using a Telegram bot.

## Supported Models

The following machine learning models are available for training and prediction:

- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Logistic Regression
- Support Vector Classifier (SVC)

## Prerequisites

Make sure you have the following installed:
- Python 3.x
- pip (for installing dependencies)
- Anaconda (optional, but recommended for managing environments)

### Install Dependencies

To set up the project, you need to install the required dependencies. Run the following command:

```bash
pip install -r requirements.txt
