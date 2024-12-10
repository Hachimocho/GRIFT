import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class DQNModel(nn.Module):
    """DQN model for predicting I-values based on node attributes."""
    
    def __init__(self, input_dim):
        super(DQNModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output Q-value
        
        # Move model to device
        self.to(self.device)
        
        # DQN specific parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        
    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_attribute_weights(self):
        """Extract the weights from the first layer as attribute importance."""
        return self.fc1.weight.data.cpu().numpy()
    
    def train_step(self, batch):
        """Train the DQN on a batch of experiences."""
        if len(batch) < self.batch_size:
            return 0.0
        
        # Sample random batch
        transitions = random.sample(batch, self.batch_size)
        
        # Separate batch into components and move to device
        states = torch.cat([t[0] for t in transitions]).to(self.device)
        rewards = torch.tensor([t[1] for t in transitions], dtype=torch.float32).to(self.device)
        
        # Compute Q values
        q_values = self(states)
        
        # Compute loss
        loss = F.mse_loss(q_values.squeeze(), rewards)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_i_value(self, node_features):
        """Calculate I-value as 1-Q for given node features."""
        with torch.no_grad():
            q_value = self(node_features)
            i_value = 1.0 - q_value.item()
        return i_value
