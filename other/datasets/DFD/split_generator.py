import os
import json


train_vids = os.listdir("/home/brg2890/major/preprocessed/DeepFakeDetection/train")

with open('/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/datasets/DFD/train.json', mode='w') as file:
    json.dump(train_vids, file)
        
val_vids = os.listdir("/home/brg2890/major/preprocessed/DeepFakeDetection/val")
with open('/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/datasets/DFD/val.json', mode='w') as file:
    json.dump(val_vids, file)
        
test_vids = os.listdir("/home/brg2890/major/preprocessed/DeepFakeDetection/test")
with open('/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/datasets/DFD/test.json', mode='w') as file:
    json.dump(test_vids, file)