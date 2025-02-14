import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PredictorNetwork(nn.Module):
    def __init__(self, input_size):
        super(PredictorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Outputs [reward, q_value]
        )
    
    def forward(self, x):
        return self.network(x)

class DQNIValuePredictor:
    def __init__(self, memory_size=10000, batch_size=32, learning_rate=0.001):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.input_size = None  # Will be set on first trace
        self.model = None
        self.optimizer = None
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_q = float('inf')
        self.max_q = float('-inf')
    
    def _initialize_model(self, input_size):
        self.input_size = input_size
        self.model = PredictorNetwork(input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def add_trace(self, node, prediction, reward, next_node):
        """Add a trace to memory"""
        if self.input_size is None:
            self._initialize_model(len(node.attributes))
            
        trace = {
            'attributes': torch.FloatTensor(node.attributes),
            'prediction': prediction,
            'reward': reward,
            'next_attributes': torch.FloatTensor(next_node.attributes) if next_node else None
        }
        self.memory.append(trace)
    
    def update(self):
        """Train the model on a batch of traces"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        current_states = torch.stack([trace['attributes'] for trace in batch]).to(self.device)
        rewards = torch.FloatTensor([trace['reward'] for trace in batch]).to(self.device)
        next_states = torch.stack([
            trace['next_attributes'] if trace['next_attributes'] is not None 
            else torch.zeros_like(trace['attributes']) 
            for trace in batch
        ]).to(self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            next_predictions = self.model(next_states)
            next_q_values = next_predictions[:, 1]
        
        target_q_values = rewards + 0.99 * next_q_values  # 0.99 is the discount factor
        
        # Current predictions
        predictions = self.model(current_states)
        predicted_rewards = predictions[:, 0]
        predicted_q_values = predictions[:, 1]
        
        # Compute loss
        loss_fn = nn.MSELoss()
        reward_loss = loss_fn(predicted_rewards, rewards)
        q_value_loss = loss_fn(predicted_q_values, target_q_values)
        total_loss = reward_loss + q_value_loss
        
        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update Q-value bounds
        with torch.no_grad():
            self.min_q = min(self.min_q, predicted_q_values.min().item())
            self.max_q = max(self.max_q, predicted_q_values.max().item())
    
    def predict(self, node):
        """Predict reward and I-value for a node"""
        if self.model is None:
            return random.random()  # Return random value if model not initialized
            
        with torch.no_grad():
            attributes = torch.FloatTensor(node.attributes).to(self.device)
            predictions = self.model(attributes)
            predicted_reward, predicted_q = predictions.cpu().numpy()
            
            # Compute I-value (1 - normalized Q-value)
            if self.max_q == self.min_q:
                i_value = 0.5  # Default when all Q-values are the same
            else:
                normalized_q = (predicted_q - self.min_q) / (self.max_q - self.min_q)
                i_value = 1.0 - normalized_q
                
            return i_value
    
    def predict_reward(self, node):
        """Predict only the reward for a node"""
        if self.model is None:
            return 0.0
            
        with torch.no_grad():
            attributes = torch.FloatTensor(node.attributes).to(self.device)
            predictions = self.model(attributes)
            predicted_reward = predictions[0].cpu().numpy()
            return predicted_reward