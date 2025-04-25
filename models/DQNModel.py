import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import os

class DQNModel(nn.Module):
    """DQN model for predicting I-values based on node attributes and embeddings."""
    
    def __init__(self, feature_dim, device, embedding_dim=512, compressed_embedding_dim=64):
        super(DQNModel, self).__init__()
        self.device = device

        # Embedding processor
        self.embedding_dim = embedding_dim
        self.compressed_embedding_dim = compressed_embedding_dim
        if self.embedding_dim > 0:
            self.embedding_processor = nn.Sequential(
                nn.Linear(self.embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.compressed_embedding_dim),
                nn.ReLU()
            ).to(self.device)
            combined_feature_dim = feature_dim + self.compressed_embedding_dim
        else:
            self.embedding_processor = None
            combined_feature_dim = feature_dim

        # Main network layers
        self.fc1 = nn.Linear(combined_feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output Q-value
        
        # Move main layers to device
        self.fc1.to(self.device)
        self.fc2.to(self.device)
        self.fc3.to(self.device)
        
        # DQN specific parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        
    def save_checkpoint(self, filepath):
        """Saves the DQN model and optimizer state dictionaries."""
        checkpoint = {
            'model_state_dict': self.state_dict(), # Use self.state_dict() directly
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Consider saving replay buffer if DQN manages it internally
            # 'replay_buffer': list(self.replay_buffer) 
        }
        torch.save(checkpoint, filepath)
        # print(f"DQNModel checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Loads the DQN model and optimizer state dictionaries."""
        if not os.path.exists(filepath):
            print(f"Warning: DQN Checkpoint file not found at {filepath}. Skipping load.")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict']) # Load directly into self
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load replay buffer if saved
        # if 'replay_buffer' in checkpoint:
        #     self.replay_buffer = deque(checkpoint['replay_buffer'], maxlen=self.replay_buffer.maxlen)
        self.to(self.device) # Ensure model is on the correct device after loading
        # print(f"DQNModel checkpoint loaded from {filepath}")
        
    def forward(self, x_features, x_embedding=None):
        # Ensure inputs are on correct device
        x_features = x_features.to(self.device)

        if self.embedding_processor is not None and x_embedding is not None:
            x_embedding = x_embedding.to(self.device)
            # Check for batch dimension (unsqueeze if single sample)
            if x_embedding.dim() == 1:
                x_embedding = x_embedding.unsqueeze(0)
            if x_features.dim() == 1:
                x_features = x_features.unsqueeze(0)

            processed_embedding = self.embedding_processor(x_embedding)
            # Concatenate features and processed embedding
            x_combined = torch.cat((x_features, processed_embedding), dim=1)
        else:
            # Check for batch dimension (unsqueeze if single sample)
            if x_features.dim() == 1:
                x_features = x_features.unsqueeze(0)
            x_combined = x_features

        # Pass through main network
        x = F.relu(self.fc1(x_combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_attribute_weights(self):
        """
        Extract the weights from the first layer as attribute importance.
        Note: These weights correspond to the concatenation of
        original features and the *compressed* embedding features.
        """
        return self.fc1.weight.data.cpu().numpy()
    
    def train_step(self, batch):
        """Train the DQN on a batch of experiences from replay buffer."""
        # This method assumes it's called by IValueTrainer, which manages the buffer
        # Use self.replay_buffer if DQN manages its own buffer internally
        if len(batch) < self.batch_size:
            # If the provided batch is smaller than batch_size, sample from internal buffer
            if len(self.replay_buffer) < self.batch_size:
                return 0.0
            transitions = random.sample(self.replay_buffer, self.batch_size)
        else:
            # If the provided batch is large enough, use it directly
            transitions = random.sample(batch, self.batch_size)
        
        # Separate batch into components
        # Assumes buffer stores tuples: (features, embedding, reward)
        state_features = torch.stack([t[0] for t in transitions])
        # Handle potential None embeddings gracefully
        state_embeddings = torch.stack([t[1] if t[1] is not None else torch.zeros(self.embedding_dim) for t in transitions])
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32).unsqueeze(1)
        
        # Move tensors to device (forward pass will also handle this, but good practice)
        state_features = state_features.to(self.device)
        state_embeddings = state_embeddings.to(self.device)
        rewards = rewards.to(self.device)
        
        # Compute Q values for current states
        q_values = self(state_features, state_embeddings)
        
        # Compute loss
        loss = F.mse_loss(q_values, rewards)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_i_value(self, node_features, node_embedding=None):
        """Predict the Q-value for given node features and embedding.

        Args:
            node_features (torch.Tensor): Tensor of node features.
            node_embedding (torch.Tensor, optional): Tensor of node embedding. Defaults to None.

        Returns:
            torch.Tensor: The predicted Q-value as a tensor.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            # Ensure inputs are on the correct device
            node_features = node_features.to(self.device)
            if node_embedding is not None:
                node_embedding = node_embedding.to(self.device)
            elif self.embedding_dim > 0:
                # Create zero tensor for embedding if needed, on the correct device
                node_embedding = torch.zeros(node_features.shape[0], self.embedding_dim, device=self.device)
            else:
                # Handle case where embedding_dim is 0 but embedding tensor might still be passed (e.g., None)
                node_embedding = None # Ensure it's explicitly None if dim is 0

            # Get Q-value from the model's forward pass
            q_value = self(node_features, node_embedding)
            
        self.train() # Set model back to training mode
        # Return the raw Q-value tensor
        return q_value
