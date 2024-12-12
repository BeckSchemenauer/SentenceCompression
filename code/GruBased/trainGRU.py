import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from collections import Counter
import time
import pickle
from tqdm import tqdm  # Progress bar

# Tokenize function
def tokenize(text):
    return text.split()

# Build vocabulary function
def build_vocab(sentences):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))
    vocab = {word: idx + 4 for idx, (word, _) in enumerate(counter.items())}  # Start index from 4
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown token
    vocab['<SOS>'] = 2  # Start of sequence token
    vocab['<EOS>'] = 3  # End of sequence token
    return vocab


# Function to save vocabulary
def save_vocab(src_vocab, trg_vocab, src_vocab_filename='src_vocab.pkl', trg_vocab_filename='trg_vocab.pkl'):
    with open(src_vocab_filename, 'wb') as f:
        pickle.dump(src_vocab, f)
    print(f"Source vocabulary saved to {src_vocab_filename}")
    
    with open(trg_vocab_filename, 'wb') as f:
        pickle.dump(trg_vocab, f)
    print(f"Target vocabulary saved to {trg_vocab_filename}")

# Dataset class for Seq2Seq
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

        # Add <SOS> at the start and <EOS> at the end of target sequence
        trg_tokens = [self.trg_vocab['<SOS>']] + trg_tokens + [self.trg_vocab['<EOS>']]

        # Truncate or pad sequences to the max length
        src_tokens = src_tokens[:self.max_len] + [self.src_vocab['<PAD>']] * (self.max_len - len(src_tokens))
        trg_tokens = trg_tokens[:self.max_len] + [self.trg_vocab['<PAD>']] * (self.max_len - len(trg_tokens))

        return torch.tensor(src_tokens), torch.tensor(trg_tokens)

# GRUSeq2Seq model definition
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

# Print model configuration
def print_model_config(model, src_vocab, trg_vocab):
    print("Model Configuration:")
    print(f"Input Dimension (Source Vocabulary Size): {len(src_vocab)}")
    print(f"Output Dimension (Target Vocabulary Size): {len(trg_vocab)}")
    print(f"Embedding Dimension: {model.embedding.embedding_dim}")
    print(f"Hidden Dimension: {model.encoder.hidden_size}")
    print(f"Number of Layers: {model.encoder.num_layers}")
    print(f"Dropout Rate: {model.dropout.p}")

# Train the model
def train(model, train_loader, optimizer, criterion, epoch, device):
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

# Evaluate the model
def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for src, trg in eval_loader:
            src, trg = src.to(device), trg.to(device)

            # Forward pass
            output = model(src, trg[:, :-1])  # Exclude the last token of target for teacher forcing
            
            # Compute loss
            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), trg[:, 1:].contiguous().view(-1))
            total_loss += loss.item()

            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

# Save the model
def save_model(model, optimizer, epoch, filename="model.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

def tune_hyperparameters():
    # Hyperparameter search space
    param_space = {
        'emb_dim': [128, 256, 512],  # Embedding dimensions
        'hidden_dim': [256, 512, 1024],  # Hidden state dimensions
        'n_layers': [1, 2, 3],  # Number of GRU layers
        'dropout': [0.2, 0.5, 0.7],  # Dropout rate
        'lr': [1e-5, 1e-4, 1e-3],  # Learning rate
        'batch_size': [32, 64],  # Batch size. Note: DO NOT try 16, will take forever
    }

    # Randomly sample hyperparameters
    best_loss = float('inf')
    best_model_params = None

    for _ in range(10):  # Number of random trials
        # Sample random hyperparameters
        emb_dim = random.choice(param_space['emb_dim'])
        hidden_dim = random.choice(param_space['hidden_dim'])
        n_layers = random.choice(param_space['n_layers'])
        dropout = random.choice(param_space['dropout'])
        lr = random.choice(param_space['lr'])
        batch_size = random.choice(param_space['batch_size'])

        # Print the current configuration
        print(f"Trial with config: emb_dim={emb_dim}, hidden_dim={hidden_dim}, n_layers={n_layers}, dropout={dropout}, lr={lr}, batch_size={batch_size}")

        # Load dataset
        train_data = pd.read_csv("Release/compressionhistory.tsv", sep='\t', on_bad_lines='warn')
        train_data["Source"] = train_data["Source"].astype(str)
        train_data["Shortening"] = train_data["Shortening"].astype(str)
        
        # Organize and split dataset as before
        dic = {}
        for i, sent in enumerate(train_data["Source"]):
            if sent in dic:
                dic[sent].append(train_data["Shortening"][i])
            else:
                dic[sent] = [train_data["Shortening"][i]]

        for i in dic.keys():
            dic[i] = sorted(dic[i], key=len)

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

        # Build vocabularies
        src_vocab = build_vocab(train_df['NewSource'])
        trg_vocab = build_vocab(train_df['NewShortening'])

        # Create DataLoader with the sampled batch_size
        train_dataset = Seq2SeqDataset(train_df, src_vocab, trg_vocab)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        eval_dataset = Seq2SeqDataset(eval_df, src_vocab, trg_vocab)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        # Model configuration with the sampled hyperparameters
        input_dim = len(src_vocab)   # Number of unique words in source vocabulary
        output_dim = len(trg_vocab)  # Number of unique words in target vocabulary

        # Initialize the model
        model = GRUSeq2Seq(input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout)

        # Move the model to the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<PAD>'])

        # Train the model for a few epochs
        epochs = 5
        for epoch in range(epochs):
            train_loss, epoch_time = train(model, train_loader, optimizer, criterion, epoch, device)
            eval_loss = evaluate(model, eval_loader, criterion, device)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Time: {epoch_time:.2f} seconds')

        # Check if the current model is better than the best one
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model_params = {
                'emb_dim': emb_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'dropout': dropout,
                'lr': lr,
                'batch_size': batch_size
            }
            print(f"New best model with Eval Loss: {best_loss:.4f}")

    print("Best model configuration:", best_model_params)
    return best_model_params

# Main function
def main():
    # Load dataset
    train_data = pd.read_csv("Release/compressionhistory.tsv", sep='\t', on_bad_lines='warn')
    train_data["Source"] = train_data["Source"].astype(str)
    train_data["Shortening"] = train_data["Shortening"].astype(str)
    
    # Organize data as required
    dic = {}
    for i, sent in enumerate(train_data["Source"]):
        if sent in dic:
            dic[sent].append(train_data["Shortening"][i])
        else:
            dic[sent] = [train_data["Shortening"][i]]

    for i in dic.keys():
        dic[i] = sorted(dic[i], key=len)

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

    # Build vocabularies
    src_vocab = build_vocab(train_df['NewSource'])
    trg_vocab = build_vocab(train_df['NewShortening'])

    # Save vocabularies
    save_vocab(src_vocab, trg_vocab)

    # Create DataLoader
    train_dataset = Seq2SeqDataset(train_df, src_vocab, trg_vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    eval_dataset = Seq2SeqDataset(eval_df, src_vocab, trg_vocab)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # Model configuration
    input_dim = len(src_vocab)   # Number of unique words in source vocabulary
    output_dim = len(trg_vocab)  # Number of unique words in target vocabulary
    emb_dim = 256
    hidden_dim = 512
    n_layers = 3
    dropout = 0.5
    lr = 1e-4

    # Initialize the model
    model = GRUSeq2Seq(input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout)

    # Print the model configuration
    print_model_config(model, src_vocab, trg_vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<PAD>'])

    # Train for a few epochs
    epochs = 10
    for epoch in range(epochs):
        train_loss, epoch_time = train(model, train_loader, optimizer, criterion, epoch, device)
        eval_loss = evaluate(model, eval_loader, criterion, device)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Time: {epoch_time:.2f} seconds')
        
        # Save model at the end of each epoch
        save_model(model, optimizer, epoch)

# Run the main function
if __name__ == "__main__":
    # for normal training, uncomment main(). For hyperparameter tuning, comment out main() and uncomment best params

    main()
    #best_params = tune_hyperparameters()
