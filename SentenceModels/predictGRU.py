import torch
from trainGRU import GRUSeq2Seq, tokenize  # Import the model definition and tokenizer
import torch.optim as optim

# Function to load the trained model
def load_model(model, optimizer, filename="model.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {filename}, starting from epoch {epoch + 1}")
    return model, optimizer, epoch

# Function to generate predictions
def generate_prediction(model, sentence, src_vocab, trg_vocab, device, max_len=50):
    model.eval()
    
    # Tokenize input sentence and convert to tensor
    src_tokens = [src_vocab.get(word, src_vocab['<UNK>']) for word in tokenize(sentence)]
    src_tokens = src_tokens[:max_len] + [src_vocab['<PAD>']] * (max_len - len(src_tokens))
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through encoder
        embedded_src = model.embedding(src_tensor)
        encoder_outputs, hidden = model.encoder(embedded_src)

        # Start the target sequence with <SOS> token
        trg_tokens = [trg_vocab['<SOS>']]  
        for _ in range(max_len):
            trg_tensor = torch.tensor(trg_tokens).unsqueeze(0).to(device)
            embedded_trg = model.embedding(trg_tensor)
            decoder_outputs, hidden = model.decoder(embedded_trg, hidden)
            output = model.fc_out(decoder_outputs)

            # Get the index of the most probable next token
            next_token = output.argmax(dim=-1)[:, -1].item()

            # Stop if <EOS> token is generated
            if next_token == trg_vocab['<EOS>']:
                break

            # Append the predicted token to target tokens
            trg_tokens.append(next_token)

    # Decode the tokens back to words (excluding <SOS> token)
    decoded_sentence = ' '.join([word for idx in trg_tokens if idx != trg_vocab['<SOS>'] 
                                 for word, word_idx in trg_vocab.items() if word_idx == idx])

    return decoded_sentence


# Load vocabularies from the training data (assuming vocab is saved as pickle files or passed as arguments)
def load_vocab(src_vocab_path, trg_vocab_path):
    import pickle
    with open(src_vocab_path, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(trg_vocab_path, 'rb') as f:
        trg_vocab = pickle.load(f)
    return src_vocab, trg_vocab

# Main function to test the model with an example sentence
def main():
    # Set device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocabularies (You should save and load these after training)
    src_vocab, trg_vocab = load_vocab('src_vocab.pkl', 'trg_vocab.pkl')

    # Initialize the model and optimizer
    input_dim = len(src_vocab)
    output_dim = len(trg_vocab)
    emb_dim = 256
    hidden_dim = 512
    n_layers = 2
    dropout = 0.2
    
    model = GRUSeq2Seq(input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters())

    # Load the trained model
    model, optimizer, epoch = load_model(model, optimizer, "model.pth")

    # Test with an example sentence
    input_sentence = "The quick brown fox jumps over the lazy dog in the quiet meadow, while the birds sing softly in the trees above."
    predicted_sentence = generate_prediction(model, input_sentence, src_vocab, trg_vocab, device)
    input2 = "Just because rural Chinese and African people may not need high dairy consumption to be healthy does not mean that Americans should be discouraged from drinking milk."
    pred2 = generate_prediction(model, input2, src_vocab, trg_vocab, device)

    # Print results
    print(f"Input: {input_sentence}")
    print(f"Predicted: {predicted_sentence}")
    print("Sample from eval set:\n")
    print(f"Input: {input2}")
    print(f"Predicted: {pred2}")

if __name__ == '__main__':
    main()
