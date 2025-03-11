import gymnasium as gym
import torch.optim as optim
import numpy as np
from torch.nn.functional import cosine_similarity
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class SentenceCompressionEnv(gym.Env):
    def __init__(self, sentence, bart_model='../modelsMSData', gpt2_model='gpt2'):
        super(SentenceCompressionEnv, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(bart_model)
        self.model = BartForConditionalGeneration.from_pretrained(bart_model)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.sentence = sentence
        self.tokens = self.tokenizer(sentence, return_tensors="pt")['input_ids'].squeeze(0)
        self.reset()

    def reset(self):
        self.current_position = 0
        self.selected_tokens = []
        return self._get_observation()

    def _get_observation(self):
        if self.current_position < len(self.tokens):
            token_id = self.tokens[self.current_position].unsqueeze(0)
            embedding = self.model.model.shared(token_id).squeeze(0)
            return embedding
        else:
            return None

    def step(self, action):
        done = False
        reward = 0
        if action == 1:
            self.selected_tokens.append(self.tokens[self.current_position].item())

        self.current_position += 1

        if self.current_position >= len(self.tokens):
            done = True
            reward = self._calculate_reward()

        observation = self._get_observation() if not done else None
        return observation, reward, done, {}

    # Combines reward values
    def _calculate_reward(self):
        compressed_sentence = self.tokenizer.decode(self.selected_tokens, skip_special_tokens=True)

        # Heavily penalize empty or nearly empty sents
        if len(compressed_sentence.strip().split()) < 2:
            return -2.0

        compression_ratio = len(self.selected_tokens) / len(self.tokens)
        fluency_score = self._calculate_fluency_score(compressed_sentence)
        retention_score = self._calculate_retention_score()
        penalty = self.apply_compression_penalty(compression_ratio)
        normalized_fluency = 1 / (1 + fluency_score)
        reward = normalized_fluency - 2 * penalty + retention_score

        return reward

    # Measures how grammatical the sentence is
    def _calculate_fluency_score(self, sentence):
        inputs = self.gpt2_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.gpt2_model(**inputs, labels=inputs["input_ids"])
        fluency_score = torch.exp(outputs.loss).item()
        return fluency_score

    # Penalty to compressions outside of target range
    def apply_compression_penalty(self, compression_ratio, target_range=(0.65, 0.85)):
        lower_target, upper_target = target_range
        if compression_ratio < lower_target:
            penalty = (lower_target - compression_ratio) ** 2 * 10
        elif compression_ratio > upper_target:
            penalty = (compression_ratio - upper_target) ** 2 * 10
        else:
            penalty = 0
        return penalty

    #Calculate the cosine similarity
    def _calculate_retention_score(self):
        with torch.no_grad():
            original_embed = torch.mean(
                self.model.model.encoder(self.tokens.unsqueeze(0)).last_hidden_state, dim=1
            )
            if self.selected_tokens:
                compressed_embed = torch.mean(
                    self.model.model.encoder(torch.tensor([self.selected_tokens])).last_hidden_state, dim=1
                )
                return cosine_similarity(original_embed, compressed_embed, dim=1).item()
            else:
                return -1.0

class BARTAgent:
    # Path to pretrained BART model
    def __init__(self, bart_model='../modelsMSData', lr=1e-5):
        self.tokenizer = BartTokenizer.from_pretrained(bart_model)
        self.max_length = 128
        self.model = BartForConditionalGeneration.from_pretrained(bart_model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, sentence, compressed_sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        labels = self.tokenizer(compressed_sentence, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)['input_ids']

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# Training Loop
def train_bart(env, agent, episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.model.to(device)
    total_loss_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_loss = 0
        step_count = 0

        while not done:
            action = np.random.choice([0, 1], p=[0.3, 0.7])
            observation, _, done, _ = env.step(action)
            step_count += 1

        compressed_sentence = env.tokenizer.decode(env.selected_tokens, skip_special_tokens=True)

        print(f"Episode {episode + 1}/{episodes} - Compressed Sentence: {compressed_sentence}")

        try:
            loss = agent.train_step(env.sentence, compressed_sentence)
            total_loss += loss
        except Exception as e:
            print(f"Error during training step: {e}")
            total_loss += 0

        avg_loss = total_loss / (step_count if step_count > 0 else 1)
        total_loss_history.append(avg_loss)

        print(f"Episode {episode + 1}/{episodes}, Average Loss: {avg_loss:.4f}")

    print("Training Complete.")
    return total_loss_history


# Training test
sentence = "The Java world simply speaking did not tolerate this kind of issue to just go on and on the way that it seems to have here"
env = SentenceCompressionEnv(sentence)
agent = BARTAgent()
train_bart(env, agent)
