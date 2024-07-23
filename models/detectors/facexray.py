import torch.nn as nn
import torch
from torch.hub import load
import face_alignment
import sys
import cv2
import numpy as np

import torchvision.transforms as T
sys.path.append('/home/MAIN/bryce/major/bryce_python_workspace/DeepfakeBias/models')
from DeepFakeMask import dfl_full
TESTING_OUTPUT = '/home/MAIN/bryce/major/bryce_python_workspace/DeepfakeBias/debug'

# list('zhanghang1989/ResNeSt', force_reload=True)

class ModelOut(nn.Module):
    def __init__(self, pretrained: bool =False, finetune : bool =False, exclude_top : bool = False,
                    output_classes: int = 3, classification_strategy: str = 'categorical', configuration: str = 'default'
        ):
        super(ModelOut, self).__init__()
        self.transform = T.ToPILImage()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda', face_detector="sfd")
        
    
    def forward(self, x):
        results = []
        print(x.shape[0])
        batch_lms = self.fa.get_landmarks_from_batch(x)
        for i, lms in zip(range(x.shape[0]), batch_lms):
        #x = x.squeeze(0)
        #print(x.shape)
        #cv2.imwrite(TESTING_OUTPUT + "/" + str(hash(x)) + ".jpeg", x)
        #image = self.transform(x)
            #print("yo")
            image = x[torch.arange(x.shape[0]), i]
            #print(image)
            #image.save(TESTING_OUTPUT + "/" + str(hash(x)) + ".jpeg")
            #cv2.imwrite(TESTING_OUTPUT + "/" + str(hash(x)) + ".jpeg", image)
            #lms = self.fa.get_landmarks_from_image(image)
            #print(type(x))
            # for z, img in enumerate(x):
            
            # for i, img_lms in enumerate(lms):
                #print(type(x[i]))
                # cv2.imwrite(TESTING_OUTPUT + "/" + str(hash(x)) + ".jpeg", np.ndarray(x))
                #print(lms)
                #print(img_lms)
            if not lms:
                print("No lms")
                results.append(0)
            else:
                mask = dfl_full(landmarks=lms.astype('int32'),face=image, channels=3).mask
                # If mask is all black, image is real
                if not mask.getbbox():
                    results.append(0)
                else:
                    results.append(1)
        
        return results

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def main():
    from torchsummary import summary
    model = ModelOut(False, False)
    summary(model.cuda(), (3,224,224))

if __name__ == "__main__":
    main()