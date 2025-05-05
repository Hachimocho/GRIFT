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

    def _process_embedding(self, embedding):
        """Processes the node embedding if the processor exists."""
        if self.embedding_processor and embedding is not None:
            # Ensure embedding is on the correct device before processing
            embedding = embedding.to(self.device)
            return self.embedding_processor(embedding)
        elif self.embedding_dim > 0:
            # Return zeros if embedding is expected but None is provided
            return torch.zeros(embedding.shape[0], self.compressed_embedding_dim, device=self.device)
        else:
            # Return None if no embedding dimension is configured
            return None

    def forward(self, node_features, node_embedding=None):
        """Forward pass to predict Q-value.

        Args:
            node_features (torch.Tensor): Tensor of node features.
            node_embedding (torch.Tensor, optional): Tensor of node embeddings. Defaults to None.

        Returns:
            torch.Tensor: Predicted Q-value.
        """
        # Ensure features are on the correct device
        node_features = node_features.to(self.device)
        
        processed_embedding = self._process_embedding(node_embedding)

        if processed_embedding is not None:
            # Ensure processed_embedding is on the correct device
            processed_embedding = processed_embedding.to(self.device)
            # print(f"DEBUG DQN Fwd: Features shape {node_features.shape}, Embedding shape {processed_embedding.shape}")
            combined_features = torch.cat((node_features, processed_embedding), dim=1)
        else:
            combined_features = node_features
        
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)  # Raw Q-value prediction
        return q_value
    
    def train_step(self, transitions):
        """Performs a single training step on a batch of transitions.

        Args:
            transitions (list): A list of tuples, where each tuple is 
                                (state_features, state_embedding, reward).

        Returns:
            float: The loss value for this training step.
        """
        if not transitions:
            return 0.0 # Or raise an error

        # Unpack the batch
        # Handle potential None embeddings carefully during stacking
        state_features_batch = torch.stack([t[0] for t in transitions]).to(self.device)
        state_embeddings_batch = torch.stack([
            t[1] if t[1] is not None 
            else torch.zeros(self.embedding_dim if self.embedding_dim > 0 else 0, device=self.device) 
            for t in transitions
        ])
         # Filter out zero tensors if no embedding dim exists
        if self.embedding_dim <= 0:
             state_embeddings_batch = None # Pass None to forward if no embeddings expected
        else:
            state_embeddings_batch = state_embeddings_batch.to(self.device)
            
        rewards_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32).unsqueeze(1).to(self.device)

        # Get current Q-value predictions from the model
        # Pass embeddings only if they exist
        if state_embeddings_batch is not None:
             current_q_values = self(state_features_batch, state_embeddings_batch)
        else:
             current_q_values = self(state_features_batch)
             
        # Calculate Loss (MSE between predicted Q and actual reward)
        # Note: Using reward directly as the target Q-value (Q(s) = R(s))
        loss = F.mse_loss(current_q_values, rewards_batch)

        # Optimize DQN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
        
    def predict_i_value(self, node_features, node_embedding=None):
        """Predict the I-value for given node features and embedding.

        Args:
            node_features (torch.Tensor): Tensor of node features.
            node_embedding (torch.Tensor, optional): Tensor of node embedding. Defaults to None.

        Returns:
            torch.Tensor: The predicted I-value as a tensor.
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
        
            # Apply sigmoid to normalize Q-value to (0, 1) range
            normalized_q_value = torch.sigmoid(q_value)

            # Calculate I-value: I = 1 - Q
            i_value = 1.0 - normalized_q_value

        self.train() # Set model back to training mode
        # Return the calculated I-value tensor
        return i_value

    def get_attribute_weights(self):
        """
        Extract the weights from the first layer as attribute importance.
        Note: These weights correspond to the concatenation of
        original features and the *compressed* embedding features.
        """
        return self.fc1.weight.data.cpu().numpy()
