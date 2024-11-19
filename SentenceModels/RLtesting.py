import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BartTokenizer, BartModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Define environment
class SentenceCompressionEnv(gym.Env):
    def __init__(self, sentence, bart_model='facebook/bart-large'):
        super(SentenceCompressionEnv, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(bart_model)
        self.model = BartModel.from_pretrained(bart_model)
        self.sentence = sentence
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokens = self.tokenizer(sentence, return_tensors="pt")['input_ids'].squeeze(0)
        self.reset()

    def reset(self):
        self.current_position = 0
        self.selected_tokens = []
        return self._get_observation()

    def _calculate_fluency_score(self, sentence):
        print(sentence)
        inputs = self.gpt2_tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.gpt2_model(**inputs, labels=inputs["input_ids"])
        fluency_score = torch.exp(outputs.loss).item()  # Lower is better
        return fluency_score

    def apply_compression_penalty(self, compression_ratio, target_range=(0.5, 0.7)):
        """
        Apply a quadratic penalty to the compression ratio deviation.
        The penalty increases significantly if the compression ratio is outside the target range (50% - 70%).
        """
        lower_target, upper_target = target_range
        if compression_ratio < lower_target:
            # Penalty increases as compression_ratio gets lower than 50%
            penalty = (lower_target - compression_ratio) ** 2 * 10  # Scaling factor
        elif compression_ratio > upper_target:
            # Penalty increases as compression_ratio gets higher than 70%
            penalty = (compression_ratio - upper_target) ** 2 * 10  # Scaling factor
        else:
            # No penalty if within the target range
            penalty = 0
        return penalty

    def _get_observation(self):
        token_id = self.tokens[self.current_position].item()
        return self.model.get_input_embeddings()(torch.tensor(token_id)).detach()

    def step(self, action):
        done = False
        reward = 0
        if action == 1:  # Keep the token
            self.selected_tokens.append(self.tokens[self.current_position].item())

        self.current_position += 1

        if self.current_position >= len(self.tokens):
            done = True
            reward = self._calculate_reward()

        observation = self._get_observation() if not done else None
        return observation, reward, done, {}

    def _calculate_reward(self):
        # Convert selected token IDs back into the compressed sentence
        compressed_sentence = self.tokenizer.decode(self.selected_tokens, skip_special_tokens=True)

        # Check if the sentence is empty or too short
        if len(compressed_sentence.strip().split()) < 2:
            return -2.0  # Penalize for empty or very short sentences

        # Calculate compression ratio: how much the sentence was reduced in length
        compression_ratio = len(self.selected_tokens) / len(self.tokens)

        # Calculate fluency score for the compressed sentence
        fluency_score = self._calculate_fluency_score(compressed_sentence)

        # Calculate retention score (similarity between the full and compressed sentences)
        retention_score = self._calculate_retention_score()

        # Apply a quadratic penalty for compression ratio deviation from target range (50% to 70%)
        penalty = self.apply_compression_penalty(compression_ratio)

        # Normalize fluency score (lower is better, so we use 1 / (1 + fluency_score))
        normalized_fluency = 1 / (1 + fluency_score)  # Higher is better

        # Reward = balance of fluency, compression ratio, and retention score with the added penalty
        reward = normalized_fluency - 2*penalty + retention_score

        # Print scores for debugging (optional)
        print("Fluency Score:", fluency_score)
        print("Compression Ratio:", compression_ratio)
        print("Retention Score:", retention_score)

        return reward

    def _calculate_retention_score(self):
        original_embed = torch.mean(self.model(self.tokens.unsqueeze(0)).last_hidden_state, dim=1)
        compressed_embed = torch.mean(self.model(torch.tensor(self.selected_tokens).unsqueeze(0)).last_hidden_state,
                                      dim=1)
        return torch.cosine_similarity(original_embed, compressed_embed).item()


# Define DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Define DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        state = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        state_action_values = self.model(state.unsqueeze(0))[0, action]
        next_state_values = 0 if done else self.model(next_state.unsqueeze(0)).max(1)[0].item()
        target_value = reward + self.gamma * next_state_values
        loss = self.criterion(state_action_values, torch.tensor(target_value))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

results = []
# Training Loop
def train(env, agent, episodes=500):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train_step(state, action, reward, next_state if not done else None, done)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        results.append((total_reward, env.tokenizer.decode(env.selected_tokens)))
        # Check for optimal reward and stop if achieved
        if total_reward >= 2:
            print("Optimal compression achieved!")
            print("Compressed Sentence:", env.tokenizer.decode(env.selected_tokens))
            break


# Usage Example
sentence = "This is an example of a long sentence that we want to compress by retaining essential information."
sentence2 = "The Java world simply speaking did not tolerate this kind of issue to just go on and on the way that it seems to have here"

env = SentenceCompressionEnv(sentence2)
agent = DQNAgent(state_dim=1024, action_dim=2)  # BART Large embedding size is 1024, action space is 2 (keep or discard)
train(env, agent)
max_tuple = max(results, key=lambda x: x[0])
print(max_tuple)
