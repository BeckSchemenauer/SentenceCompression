import json
import os

import pandas as pd
from datasets import Dataset


url = "https://github.com/google-research-datasets/sentence-compression/raw/master/data/sent-comp.train01.json.gz"
file_path = "./sent-comp.train01.json/sent-comp.train01.json"

json_data = []

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
    print(df.columns)
    print(df["graph.sentence"])
    print(df["compression.text"])
else:
    print("No valid JSON data found.")

train_data = df[['graph.sentence', 'compression.text']].dropna()
train_data.columns = ['input_text', 'target_text']
train_data = train_data[0:1000]
print(train_data)


from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token  #Use the EOS token as the padding token


# Define a preprocessing function for tokenizing
def preprocess_function(examples):
    inputs = examples["input_text"]
    outputs = examples["target_text"]

    # Tokenize inputs with padding
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    # Tokenize targets with padding and set them as labels
    labels = tokenizer(outputs, max_length=128, truncation=True, padding='max_length')

    # Assign labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_data = Dataset.from_pandas(train_data)
tokenized_datasets = train_data.map(preprocess_function, batched=True)

from transformers import Trainer, TrainingArguments
print(tokenized_datasets[0])  # Inspect the structure of the first tokenized sample

# Define the Data Collator for Seq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Start training
trainer.train()
new_drive_path = "H:/models"

os.makedirs(new_drive_path, exist_ok=True)

model.save_pretrained(new_drive_path)
tokenizer.save_pretrained(new_drive_path)