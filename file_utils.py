import json
from typing import List

import pandas as pd

from text_utils import text_processing


def read_file(f_name: str = './data/to_label.xlsx', k: int = 100, verbose: bool = False) -> List[str]:
    """
    Reads an Excel file containing messages to label and processes each message using the 'text_processing' function
    from the 'text_utils' module. It then saves the processed messages to a JSON file called 'texts.json'.

    Args:
    - f_name (str): The name or path of the Excel file to read. Default is './data/to_label.xlsx'.
    - k (int): Number of texts to sample. Default is 100
    - verbose(bool): print sampled texts or not. default value is False

    Returns:
    - texts (List[str]): List with sampled and preprocessed texts
    """
    df = pd.read_excel(f_name, index_col=0)  # ['Тип_эмоции', 'Тип_рекламы', 'companies', 'message']
    texts = list()
    for el in df.sample(k)['message'].tolist():
        new_el = text_processing(el)
        if verbose:
            print(new_el)
            print('*' * 120)
        texts.append(new_el)
    with open('texts.json', 'w') as f:
        json.dump(texts, f, indent=4, ensure_ascii=False)
    return texts
