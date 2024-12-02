from nodes.Node import Node

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
        
    def match(self, other: 'AttributeNode'):
        # Currently uses threshold averaging if there is a dispute
        if self.threshold != other.threshold:
            threshold = round((self.threshold + other.threshold) / 2)
        else:
            threshold = self.threshold
        matching = 0
        # print(type(self.attributes))
        # print(self.attributes)
        for attribute, value in self.attributes.items():
            if (attribute in other.attributes.keys()) and (other.attributes[attribute] == value):
                matching += 1
        if matching >= threshold:
            return True
        else:
            return False
        
    def __len__(self):
        return len(self.attributes)
    
    def add_attribute(self, attribute, label):
        self.attributes[label] = attribute
        
    def remove_attribute(self, label):
        if label in self.attributes.keys():
            del self.attributes[label]
        else:
            raise ValueError("Cannot remove nonexistent attribute.")
    
    
        
        