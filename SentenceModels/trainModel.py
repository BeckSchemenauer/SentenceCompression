import json
import os
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from preprocessing import getData
import pandas as pd

train_data = pd.read_json("hf://datasets/embedding-data/sentence-compression/sentence-compression_compressed.jsonl.gz", lines=True)
print(train_data["set"][0], train_data["set"][1])
train_data = train_data.iloc[0:10000]
# Max value is 20,000 for now
#train_data = getData(10000)

# Load the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

def preprocess_function(examples):
    inputs = examples["set"][0]
    outputs = examples["set"][1]

    # Tokenize inputs with padding
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    # Tokenize targets with padding and set them as labels
    labels = tokenizer(outputs, max_length=128, truncation=True, padding='max_length')

    # Assign labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_data = Dataset.from_pandas(train_data)
tokenized_datasets = train_data.map(preprocess_function)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="../results",
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    # Change this later
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()

# Change this to a drive path that can store the trained model
new_drive_path = "./modelsOtherHFData"
os.makedirs(new_drive_path, exist_ok=True)

# Saving model and tokenizer
model.save_pretrained(new_drive_path)
tokenizer.save_pretrained(new_drive_path)