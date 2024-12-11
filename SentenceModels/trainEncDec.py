import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from collections import Counter
import time
from tqdm import tqdm  # Progress bar

train_data = pd.read_csv("Release/compressionhistory.tsv", sep='\t', on_bad_lines='warn')

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

# Split the dataset into train and evaluation sets
train_df, eval_df = train_test_split(train_data, test_size=0.2, random_state=42)

# Tokenize the text and build vocab
def tokenize(text):
    return text.split()

def build_vocab(sentences):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.items())}
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown token
    return vocab

src_vocab = build_vocab(train_df['NewSource'])
trg_vocab = build_vocab(train_df['NewShortening'])

class Seq2SeqDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab, max_len=50):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]['NewSource']
        trg_text = self.data.iloc[idx]['NewShortening']

        src_tokens = [self.src_vocab.get(word, self.src_vocab['<UNK>']) for word in tokenize(src_text)]
        trg_tokens = [self.trg_vocab.get(word, self.trg_vocab['<UNK>']) for word in tokenize(trg_text)]

        # Truncate or pad sequences to the max length
        src_tokens = src_tokens[:self.max_len] + [self.src_vocab['<PAD>']] * (self.max_len - len(src_tokens))
        trg_tokens = trg_tokens[:self.max_len] + [self.trg_vocab['<PAD>']] * (self.max_len - len(trg_tokens))

        return torch.tensor(src_tokens), torch.tensor(trg_tokens)

# Create DataLoader
train_dataset = Seq2SeqDataset(train_df, src_vocab, trg_vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

eval_dataset = Seq2SeqDataset(eval_df, src_vocab, trg_vocab)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

class GRUSeq2Seq(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.encoder = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg):
        embedded_src = self.dropout(self.embedding(src))
        encoder_outputs, hidden = self.encoder(embedded_src)
        
        embedded_trg = self.dropout(self.embedding(trg))
        decoder_outputs, _ = self.decoder(embedded_trg, hidden)
        
        output = self.fc_out(decoder_outputs)
        return output

# Model configuration
input_dim = len(src_vocab)   # Number of unique words in source vocabulary
output_dim = len(trg_vocab)  # Number of unique words in target vocabulary
emb_dim = 256
hidden_dim = 512
n_layers = 2
dropout = 0.5

# Initialize the model
model = GRUSeq2Seq(input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<PAD>'])

# Training loop
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()  # Start timing the epoch
    
    # Progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for batch_idx, (src, trg) in enumerate(pbar):
            src, trg = src.to(device), trg.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])  # Exclude the last token of target for teacher forcing
            
            # Compute loss
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)  # Shift target sequence
            loss = criterion(output, trg)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update the progress bar with the average loss
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=avg_loss)
    
    # Time the epoch
    epoch_time = time.time() - start_time
    return total_loss / len(train_loader), epoch_time


# Accuracy function
def calculate_accuracy(output, trg):
    # Assuming output is of shape [batch_size, seq_len, output_dim]
    if output.dim() != 3:
        raise ValueError(f"Expected output to have 3 dimensions, but got {output.dim()} dimensions")
    
    _, pred_tokens = output.max(dim=2)  # Take the max along the output_dim
    correct = (pred_tokens == trg).float()  # Compare predictions with targets
    accuracy = correct.sum() / correct.numel()  # Calculate the average accuracy
    return accuracy.item()


# Evaluation function
def evaluate(model, eval_loader, criterion):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        for src, trg in eval_loader:
            src, trg = src.to(device), trg.to(device)

            # Forward pass
            output = model(src, trg[:, :-1])  # Exclude the last token of target for teacher forcing
            
            # Calculate accuracy before reshaping output
            accuracy = calculate_accuracy(output, trg[:, 1:])  # Use the shifted target sequence
            total_accuracy += accuracy

            # Compute loss
            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), trg[:, 1:].contiguous().view(-1))
            total_loss += loss.item()

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    return avg_loss, avg_accuracy


# Save model function
def save_model(model, optimizer, epoch, filename="model.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")


# Train for a few epochs
print("training...\n")

epochs = 10
total_start_time = time.time()  # Start timing total training
for epoch in range(epochs):
    train_loss, epoch_time = train(model, train_loader, optimizer, criterion, epoch)
    eval_loss, eval_accuracy = evaluate(model, eval_loader, criterion)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}, Time: {epoch_time:.2f} seconds')
    
    # Save model at the end of each epoch
    save_model(model, optimizer, epoch)

total_end_time = time.time()  # End timing total training
total_training_time = total_end_time - total_start_time
print(f'Total Training Time: {total_training_time:.2f} seconds')