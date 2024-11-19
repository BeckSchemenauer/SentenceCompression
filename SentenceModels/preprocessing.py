import json
import pandas as pd


url = "https://github.com/google-research-datasets/sentence-compression/raw/master/data/sent-comp.train01.json.gz"
file_path = "sent-comp.train01.json/sent-comp.train01.json"

json_data = []

def getData(numSamples):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    json_strings = content.split('\n\n')
    for json_string in json_strings:
        cleaned_content = json_string.replace('\n', '')

        if cleaned_content:
            try:
                json_objects = json.loads(f"[{cleaned_content}]")
                json_data.extend(json_objects)
            except json.JSONDecodeError as e:
                print(f"Failed to parse cleaned JSON: {e}")

    # Convert to DataFrame
    if json_data:
        df = pd.json_normalize(json_data)
    else:
        print("No valid JSON data found.")

    train_data = df[['graph.sentence', 'compression.text']].dropna()
    train_data.columns = ['input_text', 'target_text']
    return train_data[0:numSamples]
