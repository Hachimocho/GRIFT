from trainers.Trainer import Trainer
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
import random

"""
The idea of I value traversal is as follows:
1. Initialize DQN to predict Q values for nodes based on their attributes
2. Traverse primary model to nodes and generate predictions
3. Use prediction correctness as reward signal for DQN
4. DQN predicts Q values for nearby nodes to guide traversal
5. Calculate I values as 1-Q for exploration
6. Use DQN weights and prediction patterns to measure and correct both
   inter-attribute and intra-attribute bias
"""

class AttributeMetadata:
    def __init__(self, name, attr_type, possible_values=None):
        self.name = name
        self.attr_type = attr_type  # 'categorical' or 'continuous'
        self.possible_values = possible_values  # For categorical attributes
        self.value_counts = defaultdict(int)  # Track distribution of values
        self.predictions = defaultdict(list)  # Track predictions per value

class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output Q-value
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Q-value between 0 and 1
    
    def get_attribute_weights(self):
        """Extract the weights from the first layer as attribute importance."""
        return torch.abs(self.fc1.weight).mean(dim=0)

class AttributeBiasLoss(nn.Module):
    def __init__(self, attribute_metadata):
        super(AttributeBiasLoss, self).__init__()
        self.attribute_metadata = attribute_metadata
        
    def forward(self, attribute_weights, prediction_stats):
        losses = []
        
        # Inter-attribute bias (weight variance between attributes)
        mean_weight = torch.mean(attribute_weights)
        weight_variance = torch.mean((attribute_weights - mean_weight) ** 2)
        losses.append(weight_variance)
        
        # Intra-attribute bias (prediction variance within attributes)
        for attr in self.attribute_metadata:
            if attr.attr_type == 'categorical':
                # Calculate variance in prediction accuracy across different values
                accuracies = []
                for value in attr.possible_values:
                    preds = prediction_stats[attr.name][value]
                    if preds:  # If we have predictions for this value
                        accuracy = sum(preds) / len(preds)
                        accuracies.append(accuracy)
                
                if accuracies:  # If we have any accuracies
                    accuracies = torch.tensor(accuracies)
                    mean_acc = torch.mean(accuracies)
                    acc_variance = torch.mean((accuracies - mean_acc) ** 2)
                    losses.append(acc_variance)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0).cuda()

class IValueTrainer(Trainer):
    """
    IValueTrainer is a subclass of Trainer that uses DQN to predict I-values
    for efficient graph traversal while maintaining both inter-attribute and
    intra-attribute balance.
    """
    tags = ["i-value"]
    
    def __init__(self, graphmanager, train_traversal, test_traversal, models, attribute_metadata):
        super().__init__(graphmanager, train_traversal, test_traversal, models)
        
        # Store attribute metadata
        self.attribute_metadata = [
            AttributeMetadata(
                name=attr['name'],
                attr_type=attr['type'],
                possible_values=attr['possible_values']
            )
            for attr in attribute_metadata
        ]
        
        # Get sample node to determine attribute dimension
        sample_node = next(iter(self.graphmanager.get_graph().get_nodes()))
        self.input_dim = len(self.get_node_features(sample_node))
        
        # Initialize DQN for each model
        self.dqns = []
        self.dqn_optimizers = []
        self.replay_buffers = []
        self.batch_size_dqn = 32
        self.gamma = 0.99
        
        # Bias measurement and correction
        self.bias_loss = AttributeBiasLoss(self.attribute_metadata).cuda()
        self.bias_weight = 0.1  # Weight for bias loss term
        
        # Track prediction statistics per attribute value
        self.prediction_stats = defaultdict(lambda: defaultdict(list))
        
        for _ in self.models:
            dqn = DQN(self.input_dim).cuda()
            self.dqns.append(dqn)
            self.dqn_optimizers.append(torch.optim.Adam(dqn.parameters(), lr=0.001))
            self.replay_buffers.append(deque(maxlen=10000))
    
    def update_prediction_stats(self, node, correct, model_idx):
        """Update prediction statistics for each attribute value."""
        node_attrs = node.get_attributes()
        for attr in self.attribute_metadata:
            if attr.name in node_attrs:
                value = node_attrs[attr.name]
                if attr.attr_type == 'categorical':
                    self.prediction_stats[f'model_{model_idx}_{attr.name}'][value].append(float(correct))
    
    def get_node_features(self, node):
        """Extract relevant features from node attributes for DQN input."""
        features = []
        node_attrs = node.get_attributes()
        
        for attr in self.attribute_metadata:
            if attr.name in node_attrs:
                value = node_attrs[attr.name]
                if attr.attr_type == 'categorical':
                    # For categorical attributes, we might want to one-hot encode
                    # but for now we'll use the raw value
                    features.append(float(value))
                else:  # continuous
                    features.append(float(value))
            else:
                features.append(0.0)  # Default value if attribute is missing
                
        return torch.tensor(features, dtype=torch.float32)
    
    def get_attribute_bias_score(self, model_idx):
        """Measure both inter-attribute and intra-attribute bias."""
        attribute_weights = self.dqns[model_idx].get_attribute_weights()
        return self.bias_loss(attribute_weights, self.prediction_stats[f'model_{model_idx}'])
    
    def train_dqn(self, model_idx):
        """Train DQN using experience replay with comprehensive bias correction."""
        if len(self.replay_buffers[model_idx]) < self.batch_size_dqn:
            return
        
        # Sample random batch from replay buffer
        batch = random.sample(self.replay_buffers[model_idx], self.batch_size_dqn)
        state = torch.stack([item[0] for item in batch]).cuda()
        reward = torch.tensor([item[1] for item in batch], dtype=torch.float32).cuda()
        next_state = torch.stack([item[2] for item in batch]).cuda()
        
        # Compute Q values
        current_q = self.dqns[model_idx](state)
        next_q = self.dqns[model_idx](next_state).detach()
        target_q = reward + self.gamma * next_q
        
        # Add comprehensive bias correction term
        bias_score = self.get_attribute_bias_score(model_idx)
        
        # Total loss is Q-learning loss plus bias penalty
        q_loss = F.mse_loss(current_q, target_q)
        total_loss = q_loss + self.bias_weight * bias_score
        
        self.dqn_optimizers[model_idx].zero_grad()
        total_loss.backward()
        self.dqn_optimizers[model_idx].step()
        
        return bias_score.item()
    
    def get_i_value(self, node, model_idx):
        """Calculate I-value as 1-Q for a given node."""
        with torch.no_grad():
            features = self.get_node_features(node).cuda()
            q_value = self.dqns[model_idx](features.unsqueeze(0))
            return 1.0 - q_value.item()
    
    def process_node_data(self):
        """Process node data and update DQN replay buffer with comprehensive bias awareness."""
        bias_scores = []
        for i, model in enumerate(self.models):
            model_bias_scores = {'inter': [], 'intra': {}}
            
            for node, predictions in self.stored_prediction_accuracy[i].items():
                if not predictions:
                    continue
                
                # Calculate prediction accuracy
                correct_count = sum(1 for pred, label in predictions 
                                  if (pred.item() > 0.5 and label == 1) or 
                                     (pred.item() <= 0.5 and label == 0))
                accuracy = correct_count / len(predictions) if predictions else 0
                
                # Update prediction statistics for each attribute value
                self.update_prediction_stats(node, accuracy, i)
                
                # Calculate comprehensive bias score
                bias_score = self.get_attribute_bias_score(i)
                bias_scores.append(bias_score.item())
                
                # Adjust reward based on both inter and intra-attribute bias
                reward = accuracy * (1.0 - self.bias_weight * bias_score.item())
                
                # Get current node features
                current_features = self.get_node_features(node)
                
                # Get features of neighboring nodes for next state
                neighbors = self.graphmanager.get_graph().get_neighbors(node)
                if neighbors:
                    next_node = random.choice(list(neighbors))
                    next_features = self.get_node_features(next_node)
                else:
                    next_features = current_features
                
                # Add experience to replay buffer
                self.replay_buffers[i].append((current_features, reward, next_features))
                
                # Train DQN
                bias_score = self.train_dqn(i)
                
                # Update node's I-value based on DQN prediction
                i_value = self.get_i_value(node, i)
                node.set_attribute(f'i_value_model_{i}', i_value)
                
                # Clear stored predictions
                self.stored_prediction_accuracy[i][node] = []
            
            # Log bias statistics
            if bias_scores:
                avg_bias = sum(bias_scores) / len(bias_scores)
                print(f"\nModel {i} Bias Statistics:")
                print(f"Overall bias score: {avg_bias:.4f}")
                
                # Log per-attribute prediction statistics
                for attr in self.attribute_metadata:
                    if attr.attr_type == 'categorical':
                        print(f"\n{attr.name} prediction accuracies:")
                        stats = self.prediction_stats[f'model_{i}_{attr.name}']
                        for value in attr.possible_values:
                            preds = stats[value]
                            if preds:
                                acc = sum(preds) / len(preds)
                                print(f"  Value {value}: {acc:.4f}")
    
    def train(self):
        self.train_traversal.traverse()
        # For each model, add the current node to the batch
        for i in range(len(self.batches)):
            self.batches[i].append(self.train_traversal.get_pointers()[i]['current_node'])
        
        for i, [model, optim, loss] in enumerate(zip(self.models, self.optims, self.losses)):
            if len(self.batches[i]) == self.batch_size:
                model.train()
                batch = [self.transform(cv2.cvtColor(subdata.get_data().load_data(), cv2.COLOR_BGR2RGB)) for subdata in self.batches[i]]
                y_hat = model(torch.stack(batch).cuda())
                y = torch.tensor([subdata.get_label() for subdata in self.batches[i]]).unsqueeze(1).cuda()
                acc = self.train_acc.update(y_hat, y)
                self.train_acc_history.append(acc)
                # f1 = self.train_f1.update(y_hat, y)
                # auroc = self.train_auroc.update(y_hat, y)
                train_loss = loss(y_hat, y.float())
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                
                for subdata, prediction in zip(self.batches[i], y_hat):
                    self.stored_prediction_accuracy[i][subdata].append((prediction, subdata.get_label()))
                
                self.batches[i] = []
                
            elif len(self.batches[i]) > self.batch_size:
                raise ValueError("Batch size too large")
        
        return

    def val(self):
        """
        Validation phase using I-value based traversal.
        Similar to training but without updating model weights.
        """
        self.val_traversal.traverse()
        
        for i in range(len(self.batches)):
            self.batches[i].append(self.val_traversal.get_pointers()[i]['current_node'])
        
        for i, [model, _, loss] in enumerate(zip(self.models, self.optims, self.losses)):
            if len(self.batches[i]) == self.batch_size:
                model.eval()
                with torch.no_grad():
                    batch = [self.transform(cv2.cvtColor(subdata.get_data().load_data(), cv2.COLOR_BGR2RGB)) 
                            for subdata in self.batches[i]]
                    y_hat = model(torch.stack(batch).cuda())
                    y = torch.tensor([subdata.get_label() for subdata in self.batches[i]]).unsqueeze(1).cuda()
                    
                    acc = self.val_acc.update(y_hat, y)
                    self.val_acc_history[i].append(acc)
                    
                    # Store predictions for I-value updates via DQN
                    for subdata, prediction in zip(self.batches[i], y_hat):
                        self.stored_prediction_accuracy[i][subdata].append((prediction, subdata.get_label()))
                
                self.batches[i] = []
            
            elif len(self.batches[i]) > self.batch_size:
                raise ValueError("Batch size too large")

    def test(self):
        """
        Testing phase using I-value based traversal.
        Similar to validation but using the test traversal.
        """
        self.test_traversal.traverse()
        
        for i in range(len(self.batches)):
            self.batches[i].append(self.test_traversal.get_pointers()[i]['current_node'])
        
        for i, [model, _, loss] in enumerate(zip(self.models, self.optims, self.losses)):
            if len(self.batches[i]) == self.batch_size:
                model.eval()
                with torch.no_grad():
                    batch = [self.transform(cv2.cvtColor(subdata.get_data().load_data(), cv2.COLOR_BGR2RGB)) 
                            for subdata in self.batches[i]]
                    y_hat = model(torch.stack(batch).cuda())
                    y = torch.tensor([subdata.get_label() for subdata in self.batches[i]]).unsqueeze(1).cuda()
                    
                    acc = self.test_acc.update(y_hat, y)
                    self.test_acc_history[i].append(acc)
                    
                    # Store predictions for final analysis
                    for subdata, prediction in zip(self.batches[i], y_hat):
                        self.stored_prediction_accuracy[i][subdata].append((prediction, subdata.get_label()))
                
                self.batches[i] = []
            
            elif len(self.batches[i]) > self.batch_size:
                raise ValueError("Batch size too large")
