import os
from datasets.Dataset import Dataset

class FakeAVCelebDataset(Dataset):
    tags = ["deepfakes"]
    hyperparameters = None

    def load(self):
        data_root = self.data_root
        
        for sub_folder in ["FakeVideoFakeAudio", "RealVideoRealAudio", "RealVideoFakeAudio", "FakeVideoRealAudio"]:
            for label_folder in ["train", "val", "test"]:
                folder_path = os.path.join(data_root, sub_folder, label_folder)
                for video in os.listdir(folder_path):
                    video_path = os.path.join(folder_path, video)
                    self.nodes.append(self.node_class(label_folder, self.data_class(video_path, **self.data_args), [], 0 if "RealVideo" in sub_folder else 1, **self.node_args))
        
        return self.nodes