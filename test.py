import requests
import gzip
import json

# URL of the gzipped JSON file
url = "https://github.com/google-research-datasets/sentence-compression/raw/master/data/sent-comp.train01.json.gz"

# Step 1: Download the gzipped file
response = requests.get(url, stream=True)
response.raise_for_status()  # Check if the download was successful

# Step 2: Decompress and load the entire JSON content
with gzip.GzipFile(fileobj=response.raw) as f:
    file_content = f.read().decode("utf-8")

# Step 3: Split the content into possible JSON objects
# Assuming objects are separated by a newline or similar character
json_objects = file_content.splitlines()

# Step 4: Parse the first valid JSON object
first_json_obj = None
for obj in json_objects:
    if obj.strip():  # Check for non-empty lines
        try:
            first_json_obj = json.loads(obj)
            break  # Stop after the first valid JSON object
        except json.JSONDecodeError:
            continue  # Ignore invalid JSON lines

# Step 5: Extract the "graph.sentence" and "compression.text" fields if a valid object is found
if first_json_obj:
    graph_sentence = first_json_obj.get("graph", {}).get("sentence")
    compression_text = first_json_obj.get("compression", {}).get("text")

    print("Graph Sentence:", graph_sentence)
    print("Compression Text:", compression_text)
else:
    print("No valid JSON object found.")
