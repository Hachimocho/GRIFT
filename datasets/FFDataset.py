import os
from datasets.Dataset import Dataset

class FFDataset(Dataset):
    tags = ["deepfakes"]
    hyperparameters = None

    def load(self):
        data_root = self.data_root
        
        for class_folder in ["fake", "real"]:
            if class_folder == "fake":
                sub_folders = ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]
            elif class_folder == "real":
                sub_folders = ["youtube"]
            for sub_folder in sub_folders:
                for label_folder in ["train", "val", "test"]:
                    folder_path = os.path.join(data_root, class_folder, sub_folder, label_folder)
                    for video in os.listdir(folder_path):
                        video_path = os.path.join(folder_path, video)
                        self.nodes.append(self.node_class(label_folder, self.data_class(video_path, **self.data_args), [], 0 if class_folder == "real" else 1, **self.node_args))
        
        return self.nodes