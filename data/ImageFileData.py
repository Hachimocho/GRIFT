import os
import cv2
from data.Data import Data

class ImageFileData(Data):
    """
    Takes image data from a file and loads it for usage upon request.
    Low RAM overhead, high runtime impact.
    """
    
    tags = ["image", "file", "deepfakes"]
    hyperparameters = {
        "parameters": {
            "test_param": {"distribution": "uniform", "min": 0, "max": 10}
        }
    }
    supported_extensions = ["jpg", "jpeg", "png"]
    
    def __init__(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        super().__init__(indata)
        
    def set_data(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        super().set_data(indata)
        
    def load_data(self):
        return cv2.imread(self.data)