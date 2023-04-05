from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sismetanin/rubert-toxic-pikabu-2ch")
model = AutoModelForSequenceClassification.from_pretrained("sismetanin/rubert-toxic-pikabu-2ch")


# Define request body input schema
class TextRequest(BaseModel):
    text: str


def get_prediction(predictions: List[int]) -> int:
    """
    This function takes in a list of integers representing predictions for toxicity in a given input,
     and returns an integer representing whether the input is toxic or non-toxic based on a threshold of 0.5.

    Args:
        predictions: A list of integers representing the predictions for toxicity.
        Each integer in the list should be a value between 0 and 1, where values closer to 1 indicate a higher probability of toxicity.
    Returns:
    An integer representing whether the input is toxic or non-toxic based on a threshold of 0.5.
     1 indicates the input is toxic, and 0 indicates the input is non-toxic.
    """
    return int(any(map(lambda x: x > 0.5, predictions)))  # 1 if toxic 0 if non_toxic


@app.post("/predict_toxicity")
async def predict_toxicity(text_request: TextRequest) -> Dict[str, List]:
    """
    Predicts the probability of text being toxic using a sliding window approach with a pre-trained
    transformer-based model.

    Args:
        text_request: an instance of TextRequest that includes the text to be classified.

    Returns:
        A dictionary with the predicted label and the maximum probability of toxicity.

        'predictions': A binary label indicating if the text is toxic or not. 1 if toxic, 0 if non-toxic.
        'probability': The maximum probability of the text being toxic.
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
            logits = outputs.logits
            predictions.append(torch.sigmoid(logits[0][1]).item())  # append probability of toxicity
            start_indices.append(start)  # append start index of window

    # Return predictions and corresponding start indices of windows
    return {"predictions": get_prediction(predictions), 'probability': max(predictions)}
