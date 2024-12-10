import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from preprocessing import getData

# Load and preprocess dataset
train_data = pd.read_json(
    "hf://datasets/embedding-data/sentence-compression/sentence-compression_compressed.jsonl.gz",
    lines=True
)
print(train_data["set"][0], train_data["set"][1])
train_data = train_data.iloc[0:1000]

# Initialize BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Preprocessing function
def preprocess_function(examples):
    inputs = examples["set"][0]
    outputs = examples["set"][1]

    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding='max_length'
    )
    labels = tokenizer(
        outputs, max_length=128, truncation=True, padding='max_length'
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
train_data = Dataset.from_pandas(train_data)
tokenized_datasets = train_data.map(preprocess_function)

# Data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",
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
    eval_dataset=tokenized_datasets,  # For now, using train data for eval
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
output_dir = "./modelsOtherHFData"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Custom evaluation and attention visualization
def evaluate_with_attention(trainer, dataset, tokenizer, model):
    """
    Evaluate the model and visualize attention weights.
    """
    model.eval()
    example_batch = dataset[0:5]  # Take a small batch for analysis

    inputs = tokenizer(
        example_batch["set"][0],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.decoder_attentions  # Attention weights
    generated_ids = model.generate(
        **inputs, max_length=50, num_beams=4, early_stopping=True
    )
    predictions = [
        tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
    ]

    input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    output_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
    plot_attention(
        attentions[-1][0][0],
        input_tokens,
        output_tokens
    )

    print("Predictions:")
    print(predictions)

def plot_attention(attention_matrix, input_tokens, output_tokens, head=0):
    """
    Visualize attention weights for a specific attention head.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix.detach().numpy(),
        xticklabels=output_tokens,
        yticklabels=input_tokens,
        cmap="viridis"
    )
    plt.xlabel("Output Tokens")
    plt.ylabel("Input Tokens")
    plt.title(f"Attention Head {head}")
    plt.show()

# Evaluate and visualize
evaluate_with_attention(trainer, tokenized_datasets, tokenizer, model)
