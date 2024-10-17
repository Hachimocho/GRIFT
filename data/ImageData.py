from data.Data import Data
import os
import cv2

class ImageData(Data):
    """
    Takes image data from a file and stores it for usage.
    High RAM overhead, low runtime impact.
    """
    
    tags = ["image", "file"]
    supported_extensions = ["jpg", "jpeg", "png"]
    
    def __init__(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        image = cv2.imread(indata)
        super().__init__(image)
        
    def set_data(self, indata):
        assert os.path.isfile(indata) 
        assert indata.split('.')[-1] in self.supported_extensions
        image = cv2.imread(indata)
        super().set_data(image)