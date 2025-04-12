from nodes.Node import Node
import numpy as np

class AttributeNode(Node):
    # Node tags:
    tags = ["attributes", "deepfakes"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    
    def __init__(self, split, data, edges, label, attributes, threshold):
        super().__init__(split, data, edges, label)
        
        self.attributes = attributes
        self.threshold = threshold
        
    def compute_similarity(self, other: 'AttributeNode', attribute_name: str, value1, value2):
        """Compute similarity between two attribute values based on their type"""
        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            # For embeddings, use cosine similarity
            if len(value1) > 0 and len(value2) > 0:
                similarity = np.dot(value1, value2) / (np.linalg.norm(value1) * np.linalg.norm(value2))
                return similarity > 0.8  # Threshold for cosine similarity
            return False
        elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # For numeric values, check if within threshold range
            if attribute_name.startswith(('blur', 'brightness', 'contrast', 'compression')):
                return abs(value1 - value2) < 50  # Quality metric threshold
            elif attribute_name.startswith('symmetry'):
                return abs(value1 - value2) < 0.2  # Symmetry threshold
            else:
                return abs(value1 - value2) < 0.5  # Default numeric threshold
        else:
            # For boolean or categorical values, exact match
            return value1 == value2
    
    def match(self, other: 'AttributeNode'):
        """Check if two nodes match based on their attributes"""
        if not isinstance(other, AttributeNode):
            return False
            
        matching = 0
        total = 0
        
        # Compare attributes that exist in both nodes
        common_attrs = set(self.attributes.keys()) & set(other.attributes.keys())
        for attr in common_attrs:
            total += 1
            if self.compute_similarity(other, attr, self.attributes[attr], other.attributes[attr]):
                matching += 1
        
        # Use threshold as a percentage of matching attributes
        if total == 0:
            return False
        return (matching / total) >= (self.threshold / 100)
    
    def __len__(self):
        return len(self.attributes)
    
    def add_attribute(self, attribute, label):
        self.attributes[label] = attribute
        
    def remove_attribute(self, label):
        if label in self.attributes.keys():
            del self.attributes[label]
        else:
            raise ValueError("Cannot remove nonexistent attribute.")

    def __str__(self):
        return f"{self.__class__.__name__}({self.attributes})"
        