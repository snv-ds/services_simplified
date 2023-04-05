from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


app = FastAPI()

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sunny3/rubert-conversational-sentiment-balanced")
# base_model = 'sunny3/rubert-conversational-sentiment-balanced'
LABELS = ['negative', 'neutral', 'positive']
model = AutoModelForSequenceClassification.from_pretrained('sunny3/rubert-conversational-sentiment-balanced')


# Define request body input schema
class TextRequest(BaseModel):
    text: str


def get_prediction(predictions: List[List[int]]):  # 1 if positive, -1 if negative, 0 if neutral
    """
    Returns the predicted sentiment label for a given input text, based on a list of predicted sentiment scores
    obtained from a machine learning model. A sentiment label of 1 indicates a positive sentiment, -1 indicates a negative
    sentiment, and 0 indicates a neutral sentiment.

    Args:
    - predictions (list): A list containing the predicted sentiment scores for an input text. The list should have
                          shape (1, 3), where the three scores represent the probabilities of a positive, negative, and
                          neutral sentiment label, respectively.

    Returns:
    - int: The predicted sentiment label for the input text. If all three sentiment scores are below a threshold of 0.37,
           then the function returns 0 to indicate a neutral sentiment. Otherwise, the function returns the sentiment label
           with the highest score (i.e., the most probable sentiment label), adjusted to the expected range of [-1, 1]
           to indicate a negative or positive sentiment label, respectively.
    """
    if all(map(lambda x: x < 0.37, predictions[0])):  # not so obvious example
        return 0
    else:
        return torch.argmax(torch.Tensor(predictions[0])).item() - 1


@app.post("/predict_emotions")
async def predict_emotions(text_request: TextRequest):
    """
    The function is an API endpoint for predicting emotions in a given text using a sliding window technique.
    The input text is tokenized, and the resulting tokens are processed in windows to obtain predictions for each window.
    The predictions are then aggregated to provide the final prediction for the input text.
    Args: text_request: An instance of the TextRequest class representing the text to predict the emotion for.

    Returns: A dictionary containing the following keys:
        - predictions: An integer representing the predicted emotion. 1 for positive, -1 for negative, and 0 for neutral.
        - probability: A float representing the probability of the predicted emotion.
    """
    # Tokenize input text
    inputs = tokenizer(text_request.text, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Get sequence length
    seq_len = inputs["input_ids"].shape[1]

    if seq_len > 512:
        # Define sliding window parameters
        stride = int(seq_len / 2)
        window_size = seq_len
    else:
        stride = seq_len
        window_size = seq_len

    # Initialize output variables
    predictions = []
    start_indices = []

    # Perform sliding window classification
    for i in range(0, len(inputs["input_ids"][0]), stride):
        # Get current window
        start = i
        end = i + window_size
        if end > len(inputs["input_ids"][0]):
            end = len(inputs["input_ids"][0])
        window_input_ids = inputs["input_ids"][:, start:end]
        window_attention_mask = inputs["attention_mask"][:, start:end]

        # Make prediction
        with torch.no_grad():
            outputs = model(window_input_ids, attention_mask=window_attention_mask)
            predictions.append(torch.softmax(outputs.logits, -1).cpu().flatten().tolist())  # append probability of predicted emotion
            start_indices.append(start)  # append start index of window

    # Return predictions and corresponding start indices of windows
    return {"predictions": get_prediction(predictions), 'probability': max(predictions[0])}

