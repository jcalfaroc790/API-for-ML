pip install python-telegram-bot==13.7  # compatible version
import os
import requests
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Initialize the bot with your token
BOT_TOKEN = "7895780185:AAHOAR94LE5G_uZFDjaE4JVznuy0_KXrAOw"
API_BASE_URL = "http://127.0.0.1:5000"  # Replace with the base URL of your API

# Create a bot instance
bot = Bot(token=BOT_TOKEN)

# Define command handlers

def start(update: Update, context: CallbackContext):
    """Send a message when the command /start is issued."""
    update.message.reply_text("Hello! I am BonusHABot. I can help you interact with your API. Use /help to see commands.")

def help_command(update: Update, context: CallbackContext):
    """Send a message when the command /help is issued."""
    commands = """
    /start - Start the bot
    /models - List available models
    /upload - Upload data (send CSV file)
    /train <model_name> - Train a model
    /predict <model_name> - Predict using a model
    /delete <model_name> - Delete a trained model
    """
    update.message.reply_text(commands)

def list_models(update: Update, context: CallbackContext):
    """Fetch available models from the API."""
    response = requests.get(f"{API_BASE_URL}/models")
    if response.ok:
        models = response.json()
        update.message.reply_text(f"Available models: {', '.join(models)}")
    else:
        update.message.reply_text("Error fetching models.")

def train_model(update: Update, context: CallbackContext):
    """Train a model by name."""
    if context.args:
        model_name = context.args[0]
        response = requests.post(f"{API_BASE_URL}/train", json={"model_name": model_name})
        if response.ok:
            update.message.reply_text(f"{model_name} model trained successfully.")
        else:
            update.message.reply_text(f"Error training {model_name}.")
    else:
        update.message.reply_text("Please specify a model name. Usage: /train <model_name>")

def predict(update: Update, context: CallbackContext):
    """Make a prediction using a trained model."""
    if context.args:
        model_name = context.args[0]
        input_data = list(map(float, context.args[1:]))  # Example of handling input data from command arguments
        response = requests.post(f"{API_BASE_URL}/predict", json={"model_name": model_name, "input_data": input_data})
        if response.ok:
            prediction = response.json().get("prediction", [])
            update.message.reply_text(f"Prediction: {prediction}")
        else:
            update.message.reply_text(f"Error making prediction with {model_name}.")
    else:
        update.message.reply_text("Please specify a model name and input data. Usage: /predict <model_name> <data>")

def delete_model(update: Update, context: CallbackContext):
    """Delete a trained model by name."""
    if context.args:
        model_name = context.args[0]
        response = requests.delete(f"{API_BASE_URL}/delete_model", json={"model_name": model_name})
        if response.ok:
            update.message.reply_text(f"{model_name} model deleted successfully.")
        else:
            update.message.reply_text(f"Error deleting {model_name}.")
    else:
        update.message.reply_text("Please specify a model name. Usage: /delete <model_name>")

# Setting up the Updater and Dispatcher
updater = Updater(BOT_TOKEN, use_context=True)
dispatcher = updater.dispatcher

# Add command handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("help", help_command))
dispatcher.add_handler(CommandHandler("models", list_models))
dispatcher.add_handler(CommandHandler("train", train_model))
dispatcher.add_handler(CommandHandler("predict", predict))
dispatcher.add_handler(CommandHandler("delete", delete_model))

# Start the bot
print("Bot is running...")
updater.start_polling()
updater.idle()
