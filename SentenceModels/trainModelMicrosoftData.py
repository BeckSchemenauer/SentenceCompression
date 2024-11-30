import os
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
import pandas as pd

train_data = pd.read_csv("H:/Release/compressionhistory.tsv", sep='\t', on_bad_lines='warn')

train_data["Source"] = train_data["Source"].astype(str)
train_data["Shortening"] = train_data["Shortening"].astype(str)
dic = {}
for i, sent in enumerate(train_data["Source"]):
    if sent in dic:
        list = dic[sent]
        list.append(train_data["Shortening"][i])
    else:
        dic[sent] = [train_data["Shortening"][i]]

for i in dic.keys():
    list = dic[i]
    dic[i] = sorted(list, key=len)

train_data["NewSource"] = None
train_data["NewShortening"] = None
for i, sent in enumerate(dic.keys()):
    train_data.loc[i, "NewSource"] = sent
    train_data.loc[i, "NewShortening"] = dic[sent][0]

train_data.dropna(inplace=True)
train_data["NewSource"] = train_data["NewSource"].astype(str)
train_data["NewShortening"] = train_data["NewShortening"].astype(str)
train_data.drop(["Source", "Shortening"], axis=1, inplace=True)
train_data = train_data[["NewSource", "NewShortening"]]

# Load the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def preprocess_function(examples):
    inputs = examples["NewSource"]
    outputs = examples["NewShortening"]

    # Tokenize inputs with padding
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    # Tokenize targets with padding and set them as labels
    labels = tokenizer(outputs, max_length=128, truncation=True, padding='max_length')

    # Assign labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_data = Dataset.from_pandas(train_data)
tokenized_datasets = train_data.map(preprocess_function)

train, eval = train_test_split(tokenized_datasets, test_size=0.2)

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
    train_dataset=train,
    eval_dataset=eval,
    data_collator=data_collator,
)

trainer.train()

# Change this to a drive path that can store the trained model
new_drive_path = "modelsMSData"
os.makedirs(new_drive_path, exist_ok=True)

# Saving model and tokenizer
model.save_pretrained(new_drive_path)
tokenizer.save_pretrained(new_drive_path)