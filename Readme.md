# Services_simplified

This is a simple service built using FastAPI that predicts toxicity and emotions of text inputs.

## Installation

- Clone this repository.
- Navigate to the root directory of the repository.
- Install the required dependencies by running pip install -r requirements.txt.
## Usage

To run the service, navigate to the root directory of the repository and run uvicorn main:app --reload in the terminal.
This will start the service at http://localhost:8000.

Once the service is running, you can test it by sending a POST request to http://localhost:8000/predict_toxicity or 
http://localhost:8000/predict_emotions, depending on the functionality you want to test.

In order to send a POST request, you can use any tool you like. For example, you can use the requests library in Python. Here is some example code:
```python
import requests

url = 'http://localhost:8000/predict_toxicity'
data = {'text': 'This is some text to test.'}
response = requests.post(url, json=data)
print(response.json())
```

or you can use simplified version in `requester.py` file 

This will send a POST request to the predict_toxicity endpoint with the text "This is some text to test." as the input.
The response will be printed to the console.

## Documentation

To read the API documentation, navigate to http://localhost:8000/docs in your web browser while the service is running.
This will bring up the Swagger UI, which provides documentation for all the endpoints of the service.