import pandas as pd
import requests
from tqdm.auto import tqdm

from file_utils import read_file


def send_request(text,
                 url="http://localhost:8000/predict_emotions",  # for toxicity "http://localhost:8000/predict_toxicity"
                 headers={"Content-Type": "application/json", "Accept": "application/json"}):

    payload = {"text": text}

    response = requests.post(url, headers=headers, json=payload)

    if response.ok:
        result = response.json()
        return result
    else:
        print("Error:", response.status_code, response.reason)
        assert 'error getting prediction from model'
        return None


if __name__ == '__main__':
    texts = read_file(k=50)
    results = list()
    for text in tqdm(texts):
        result = {'processed_text': text}
        result.update(send_request(text))
        results.append(result)
    pd.DataFrame(results).to_csv('./toxic_results.tsv', sep='\t')
