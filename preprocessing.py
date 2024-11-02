import requests
import gzip
import json
import pandas as pd
from io import BytesIO

# URL to the raw .gz file on GitHub
url = "https://github.com/google-research-datasets/sentence-compression/raw/master/data/sent-comp.train01.json.gz"

# Step 1: Download the gzipped JSON file
response = requests.get(url)
compressed_file = BytesIO(response.content)

# Step 2: Decompress and read line by line
data = []
with gzip.open(compressed_file, 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            # Load each JSON object in the file
            entry = json.loads(line)
            # Extract the fields you need
            data.append({
                "graph_text": entry.get("graph", {}).get("text", ""),
                "compression_text": entry.get("compression", {}).get("text", "")
            })
        except json.JSONDecodeError:
            # Skip lines that aren't valid JSON objects
            continue

# Step 3: Convert to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())
