import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import math

class ResidualBlock(nn.Module):
    """Residual block with skip connections for DQN."""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)

class ResidualDQNModel(nn.Module):
    """Deep Residual DQN with skip connections and modern regularization."""
    
    def __init__(self, feature_dim, device, embedding_dim=512, 
                 hidden_sizes=[256, 256, 128, 64], dropout=0.3, 
                 compressed_embedding_dim=64):
        super(ResidualDQNModel, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.compressed_embedding_dim = compressed_embedding_dim

        # Enhanced embedding processor with convolutions
        if self.embedding_dim > 0:
            self.embedding_processor = nn.Sequential(
                nn.Linear(self.embedding_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, self.compressed_embedding_dim),
                nn.ReLU()
            ).to(self.device)
            combined_feature_dim = feature_dim + self.compressed_embedding_dim
        else:
            self.embedding_processor = None
            combined_feature_dim = feature_dim

        # Initial projection to first hidden size
        self.input_projection = nn.Linear(combined_feature_dim, hidden_sizes[0]).to(self.device)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            if hidden_sizes[i] == hidden_sizes[i + 1]:
                # Same dimensions - can use residual connection
                self.residual_blocks.append(ResidualBlock(hidden_sizes[i], hidden_sizes[i], dropout))
            else:
                # Different dimensions - use regular block with projection
                self.residual_blocks.append(nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.BatchNorm1d(hidden_sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        ).to(self.device)
        
        # DQN specific parameters
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, eta_min=1e-6
        )
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99

    def _process_embedding(self, embedding):
        """Enhanced embedding processing."""
        if self.embedding_processor and embedding is not None:
            embedding = embedding.to(self.device)
            return self.embedding_processor(embedding)
        elif self.embedding_dim > 0:
            return torch.zeros(embedding.shape[0], self.compressed_embedding_dim, device=self.device)
        else:
            return None

    def forward(self, node_features, node_embedding=None):
        """Forward pass through residual network."""
        node_features = node_features.to(self.device)
        processed_embedding = self._process_embedding(node_embedding)

        if processed_embedding is not None:
            combined_features = torch.cat((node_features, processed_embedding), dim=1)
        else:
            combined_features = node_features
        
        x = F.relu(self.input_projection(combined_features))
        
        for block in self.residual_blocks:
            x = block(x)
        
        q_value = self.output_layer(x)
        return q_value

    def train_step(self, transitions):
        """Enhanced training step with learning rate scheduling."""
        if not transitions:
            return 0.0
            
        loss = self._compute_loss(transitions)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

    def _compute_loss(self, transitions):
        """Compute training loss."""
        state_features_batch = torch.stack([t[0] for t in transitions]).to(self.device)
        state_embeddings_batch = self._process_embeddings_batch(transitions)
        rewards_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32).unsqueeze(1).to(self.device)

        if state_embeddings_batch is not None:
            current_q_values = self(state_features_batch, state_embeddings_batch)
        else:
            current_q_values = self(state_features_batch)
             
        return F.mse_loss(current_q_values, rewards_batch)

    def _process_embeddings_batch(self, transitions):
        """Process embeddings for batch training."""
        if self.embedding_dim <= 0:
            return None
            
        embeddings = []
        for t in transitions:
            if t[1] is not None:
                embeddings.append(t[1])
            else:
                embeddings.append(torch.zeros(self.embedding_dim, device=self.device))
        return torch.stack(embeddings).to(self.device)

    def predict_i_value(self, node_features, node_embedding=None):
        """Predict I-value with improved numerical stability."""
        self.eval()
        with torch.no_grad():
            node_features = node_features.to(self.device)
            if node_embedding is not None:
                node_embedding = node_embedding.to(self.device)
            elif self.embedding_dim > 0:
                node_embedding = torch.zeros(node_features.shape[0], self.embedding_dim, device=self.device)
            
            q_value = self(node_features, node_embedding)
            # Improved numerical stability for sigmoid
            normalized_q_value = torch.sigmoid(torch.clamp(q_value, -10, 10))
            i_value = 1.0 - normalized_q_value

        self.train()
        return i_value

    def save_checkpoint(self, filepath):
        """Enhanced checkpoint saving."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'embedding_dim': self.embedding_dim,
            'compressed_embedding_dim': self.compressed_embedding_dim
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Enhanced checkpoint loading."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.to(self.device)

    def get_attribute_weights(self):
        """Extract feature importance from the first layer."""
        return self.input_projection.weight.data.cpu().numpy()


class AttentionDQNModel(nn.Module):
    """Attention-based DQN using transformer encoder blocks."""
    
    def __init__(self, feature_dim, device, embedding_dim=512, 
                 embed_dim=256, num_heads=8, num_layers=3, dropout=0.3,
                 compressed_embedding_dim=64):
        super(AttentionDQNModel, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.compressed_embedding_dim = compressed_embedding_dim
        self.embed_dim = embed_dim

        # Embedding processor
        if self.embedding_dim > 0:
            self.embedding_processor = nn.Sequential(
                nn.Linear(self.embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.compressed_embedding_dim),
                nn.ReLU()
            ).to(self.device)
            combined_feature_dim = feature_dim + self.compressed_embedding_dim
        else:
            self.embedding_processor = None
            combined_feature_dim = feature_dim

        # Project input features to embedding dimension
        self.input_projection = nn.Linear(combined_feature_dim, embed_dim).to(self.device)
        
        # Positional encoding for features (treating each feature as a token)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout).to(self.device)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers).to(self.device)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        ).to(self.device)
        
        # DQN parameters
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, eta_min=1e-7
        )
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99

    def _process_embedding(self, embedding):
        """Process embeddings."""
        if self.embedding_processor and embedding is not None:
            embedding = embedding.to(self.device)
            return self.embedding_processor(embedding)
        elif self.embedding_dim > 0:
            return torch.zeros(embedding.shape[0], self.compressed_embedding_dim, device=self.device)
        else:
            return None

    def forward(self, node_features, node_embedding=None):
        """Forward pass through attention mechanism."""
        node_features = node_features.to(self.device)
        processed_embedding = self._process_embedding(node_embedding)

        if processed_embedding is not None:
            combined_features = torch.cat((node_features, processed_embedding), dim=1)
        else:
            combined_features = node_features
        
        # Project to embedding dimension and add sequence dimension
        x = self.input_projection(combined_features)  # [batch, embed_dim]
        x = x.unsqueeze(1)  # [batch, 1, embed_dim] - treat as single token sequence
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)  # [batch, 1, embed_dim]
        
        # Pool and output
        x = x.squeeze(1)  # [batch, embed_dim]
        q_value = self.output_head(x)
        
        return q_value

    def train_step(self, transitions):
        """Training step with attention-specific optimizations."""
        if not transitions:
            return 0.0
            
        loss = self._compute_loss(transitions)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)  # Smaller clip for attention
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

    def _compute_loss(self, transitions):
        """Compute loss for attention model."""
        state_features_batch = torch.stack([t[0] for t in transitions]).to(self.device)
        state_embeddings_batch = self._process_embeddings_batch(transitions)
        rewards_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32).unsqueeze(1).to(self.device)

        if state_embeddings_batch is not None:
            current_q_values = self(state_features_batch, state_embeddings_batch)
        else:
            current_q_values = self(state_features_batch)
             
        return F.mse_loss(current_q_values, rewards_batch)

    def _process_embeddings_batch(self, transitions):
        """Process embeddings batch for attention model."""
        if self.embedding_dim <= 0:
            return None
            
        embeddings = []
        for t in transitions:
            if t[1] is not None:
                embeddings.append(t[1])
            else:
                embeddings.append(torch.zeros(self.embedding_dim, device=self.device))
        return torch.stack(embeddings).to(self.device)

    def predict_i_value(self, node_features, node_embedding=None):
        """Predict I-value using attention mechanism."""
        self.eval()
        with torch.no_grad():
            node_features = node_features.to(self.device)
            if node_embedding is not None:
                node_embedding = node_embedding.to(self.device)
            elif self.embedding_dim > 0:
                node_embedding = torch.zeros(node_features.shape[0], self.embedding_dim, device=self.device)
            
            q_value = self(node_features, node_embedding)
            normalized_q_value = torch.sigmoid(torch.clamp(q_value, -10, 10))
            i_value = 1.0 - normalized_q_value

        self.train()
        return i_value

    def save_checkpoint(self, filepath):
        """Save attention model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'embedding_dim': self.embedding_dim,
            'compressed_embedding_dim': self.compressed_embedding_dim,
            'embed_dim': self.embed_dim
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Load attention model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.to(self.device)

    def get_attribute_weights(self):
        """Get attention weights for interpretability."""
        return self.input_projection.weight.data.cpu().numpy()


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class ConvEmbeddingDQN(nn.Module):
    """DQN with convolutional processing of face embeddings."""
    
    def __init__(self, feature_dim, device, embedding_dim=512, 
                 conv_channels=[64, 128, 256], kernel_size=3, dropout=0.3,
                 compressed_embedding_dim=64):
        super(ConvEmbeddingDQN, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.compressed_embedding_dim = compressed_embedding_dim

        # Convolutional embedding processor
        if self.embedding_dim > 0:
            # Reshape embedding to "image-like" format for 1D convolution
            # Treat embedding as 1D signal with multiple channels
            conv_layers = []
            in_channels = 1
            
            for out_channels in conv_channels:
                conv_layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                in_channels = out_channels
            
            # Global average pooling and final projection
            conv_layers.extend([
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(conv_channels[-1], self.compressed_embedding_dim),
                nn.ReLU()
            ])
            
            self.embedding_processor = nn.Sequential(*conv_layers).to(self.device)
            combined_feature_dim = feature_dim + self.compressed_embedding_dim
        else:
            self.embedding_processor = None
            combined_feature_dim = feature_dim

        # Main network
        self.main_network = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        ).to(self.device)
        
        # DQN parameters
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, eta_min=1e-6
        )
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99

    def _process_embedding(self, embedding):
        """Process embeddings with convolutions."""
        if self.embedding_processor and embedding is not None:
            embedding = embedding.to(self.device)
            # Reshape for 1D convolution: [batch, 1, embedding_dim]
            embedding = embedding.unsqueeze(1)
            return self.embedding_processor(embedding)
        elif self.embedding_dim > 0:
            return torch.zeros(embedding.shape[0], self.compressed_embedding_dim, device=self.device)
        else:
            return None

    def forward(self, node_features, node_embedding=None):
        """Forward pass with convolutional embedding processing."""
        node_features = node_features.to(self.device)
        processed_embedding = self._process_embedding(node_embedding)

        if processed_embedding is not None:
            combined_features = torch.cat((node_features, processed_embedding), dim=1)
        else:
            combined_features = node_features
        
        q_value = self.main_network(combined_features)
        return q_value

    def train_step(self, transitions):
        """Training step for convolutional model."""
        if not transitions:
            return 0.0
            
        loss = self._compute_loss(transitions)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

    def _compute_loss(self, transitions):
        """Compute loss for convolutional model."""
        state_features_batch = torch.stack([t[0] for t in transitions]).to(self.device)
        state_embeddings_batch = self._process_embeddings_batch(transitions)
        rewards_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32).unsqueeze(1).to(self.device)

        if state_embeddings_batch is not None:
            current_q_values = self(state_features_batch, state_embeddings_batch)
        else:
            current_q_values = self(state_features_batch)
             
        return F.mse_loss(current_q_values, rewards_batch)

    def _process_embeddings_batch(self, transitions):
        """Process embeddings batch for convolutional model."""
        if self.embedding_dim <= 0:
            return None
            
        embeddings = []
        for t in transitions:
            if t[1] is not None:
                embeddings.append(t[1])
            else:
                embeddings.append(torch.zeros(self.embedding_dim, device=self.device))
        return torch.stack(embeddings).to(self.device)

    def predict_i_value(self, node_features, node_embedding=None):
        """Predict I-value using convolutional processing."""
        self.eval()
        with torch.no_grad():
            node_features = node_features.to(self.device)
            if node_embedding is not None:
                node_embedding = node_embedding.to(self.device)
            elif self.embedding_dim > 0:
                node_embedding = torch.zeros(node_features.shape[0], self.embedding_dim, device=self.device)
            
            q_value = self(node_features, node_embedding)
            normalized_q_value = torch.sigmoid(torch.clamp(q_value, -10, 10))
            i_value = 1.0 - normalized_q_value

        self.train()
        return i_value

    def save_checkpoint(self, filepath):
        """Save convolutional model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'embedding_dim': self.embedding_dim,
            'compressed_embedding_dim': self.compressed_embedding_dim
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Load convolutional model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.to(self.device)

    def get_attribute_weights(self):
        """Get feature importance from the first layer."""
        return self.main_network[0].weight.data.cpu().numpy()


class EnsembleDQNModel(nn.Module):
    """Ensemble of multiple DQN models for robust predictions."""
    
    def __init__(self, feature_dim, device, embedding_dim=512, num_models=5,
                 compressed_embedding_dim=64, dropout=0.3):
        super(EnsembleDQNModel, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.compressed_embedding_dim = compressed_embedding_dim
        self.num_models = num_models

        # Create multiple DQN models with different architectures
        self.models = nn.ModuleList()
        
        for i in range(num_models):
            # Vary architecture slightly for each model
            hidden_size = 128 + (i * 32)  # Different sizes: 128, 160, 192, 224, 256
            
            if embedding_dim > 0:
                embedding_processor = nn.Sequential(
                    nn.Linear(embedding_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, compressed_embedding_dim),
                    nn.ReLU()
                )
                combined_feature_dim = feature_dim + compressed_embedding_dim
            else:
                embedding_processor = None
                combined_feature_dim = feature_dim
            
            # Different architectures for diversity
            if i % 3 == 0:  # Simple deep network
                model = nn.Sequential(
                    nn.Linear(combined_feature_dim, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1)
                )
            elif i % 3 == 1:  # Network with batch norm
                model = nn.Sequential(
                    nn.Linear(combined_feature_dim, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1)
                )
            else:  # Wider network
                model = nn.Sequential(
                    nn.Linear(combined_feature_dim, hidden_size * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1)
                )
            
            # Combine embedding processor and main model
            full_model = nn.ModuleDict({
                'embedding_processor': embedding_processor,
                'main_model': model
            })
            self.models.append(full_model.to(device))

        # Ensemble-specific parameters
        self.optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) 
                          for model in self.models]
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, eta_min=1e-6)
                          for opt in self.optimizers]
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99

    def _process_embedding(self, embedding, model_idx):
        """Process embedding for specific model."""
        model = self.models[model_idx]
        if model['embedding_processor'] is not None and embedding is not None:
            embedding = embedding.to(self.device)
            return model['embedding_processor'](embedding)
        elif self.embedding_dim > 0:
            return torch.zeros(embedding.shape[0], self.compressed_embedding_dim, device=self.device)
        else:
            return None

    def forward(self, node_features, node_embedding=None, model_idx=None):
        """Forward pass through ensemble (or specific model)."""
        node_features = node_features.to(self.device)
        
        if model_idx is not None:
            # Use specific model
            processed_embedding = self._process_embedding(node_embedding, model_idx)
            if processed_embedding is not None:
                combined_features = torch.cat((node_features, processed_embedding), dim=1)
            else:
                combined_features = node_features
            return self.models[model_idx]['main_model'](combined_features)
        else:
            # Use all models and average
            outputs = []
            for i in range(self.num_models):
                processed_embedding = self._process_embedding(node_embedding, i)
                if processed_embedding is not None:
                    combined_features = torch.cat((node_features, processed_embedding), dim=1)
                else:
                    combined_features = node_features
                output = self.models[i]['main_model'](combined_features)
                outputs.append(output)
            
            # Average ensemble predictions
            return torch.mean(torch.stack(outputs), dim=0)

    def train_step(self, transitions):
        """Train all models in the ensemble."""
        if not transitions:
            return 0.0
            
        total_loss = 0.0
        
        # Train each model in the ensemble
        for i in range(self.num_models):
            loss = self._compute_loss(transitions, i)
            
            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), max_norm=1.0)
            self.optimizers[i].step()
            self.schedulers[i].step()
            
            total_loss += loss.item()
        
        return total_loss / self.num_models

    def _compute_loss(self, transitions, model_idx):
        """Compute loss for specific model in ensemble."""
        state_features_batch = torch.stack([t[0] for t in transitions]).to(self.device)
        state_embeddings_batch = self._process_embeddings_batch(transitions)
        rewards_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32).unsqueeze(1).to(self.device)

        if state_embeddings_batch is not None:
            current_q_values = self(state_features_batch, state_embeddings_batch, model_idx)
        else:
            current_q_values = self(state_features_batch, model_idx=model_idx)
             
        return F.mse_loss(current_q_values, rewards_batch)

    def _process_embeddings_batch(self, transitions):
        """Process embeddings batch for ensemble."""
        if self.embedding_dim <= 0:
            return None
            
        embeddings = []
        for t in transitions:
            if t[1] is not None:
                embeddings.append(t[1])
            else:
                embeddings.append(torch.zeros(self.embedding_dim, device=self.device))
        return torch.stack(embeddings).to(self.device)

    def predict_i_value(self, node_features, node_embedding=None):
        """Predict I-value using ensemble averaging."""
        self.eval()
        with torch.no_grad():
            node_features = node_features.to(self.device)
            if node_embedding is not None:
                node_embedding = node_embedding.to(self.device)
            elif self.embedding_dim > 0:
                node_embedding = torch.zeros(node_features.shape[0], self.embedding_dim, device=self.device)
            
            q_value = self(node_features, node_embedding)  # Uses ensemble averaging
            normalized_q_value = torch.sigmoid(torch.clamp(q_value, -10, 10))
            i_value = 1.0 - normalized_q_value

        self.train()
        return i_value

    def get_prediction_uncertainty(self, node_features, node_embedding=None):
        """Get prediction uncertainty based on ensemble disagreement."""
        self.eval()
        with torch.no_grad():
            node_features = node_features.to(self.device)
            if node_embedding is not None:
                node_embedding = node_embedding.to(self.device)
            elif self.embedding_dim > 0:
                node_embedding = torch.zeros(node_features.shape[0], self.embedding_dim, device=self.device)
            
            # Get predictions from all models
            predictions = []
            for i in range(self.num_models):
                q_value = self(node_features, node_embedding, model_idx=i)
                i_value = 1.0 - torch.sigmoid(torch.clamp(q_value, -10, 10))
                predictions.append(i_value)
            
            predictions = torch.stack(predictions)
            mean_prediction = torch.mean(predictions, dim=0)
            std_prediction = torch.std(predictions, dim=0)

        self.train()
        return mean_prediction, std_prediction

    def save_checkpoint(self, filepath):
        """Save ensemble model checkpoint."""
        checkpoint = {
            'models_state_dict': [model.state_dict() for model in self.models],
            'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
            'schedulers_state_dict': [sched.state_dict() for sched in self.schedulers],
            'embedding_dim': self.embedding_dim,
            'compressed_embedding_dim': self.compressed_embedding_dim,
            'num_models': self.num_models
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Load ensemble model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for i, model in enumerate(self.models):
            model.load_state_dict(checkpoint['models_state_dict'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizers_state_dict'][i])
            if 'schedulers_state_dict' in checkpoint:
                self.schedulers[i].load_state_dict(checkpoint['schedulers_state_dict'][i])
        
        for model in self.models:
            model.to(self.device)

    def get_attribute_weights(self):
        """Get average attribute weights across ensemble."""
        all_weights = []
        for model in self.models:
            # Get first linear layer weights from main model
            first_layer = None
            for module in model['main_model'].modules():
                if isinstance(module, nn.Linear):
                    first_layer = module
                    break
            if first_layer is not None:
                all_weights.append(first_layer.weight.data.cpu().numpy())
        
        if all_weights:
            return np.mean(all_weights, axis=0)
        else:
            return None 