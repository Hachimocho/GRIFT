import wandb
from trainers.DeepfakeAttributeTrainer import DeepfakeAttributeTrainer
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from datasets.AIFaceDataset import AIFaceDataset
from data.ImageFileData import ImageFileData
from nodes.atrnode import AttributeNode


dataloader = UnclusteredDeepfakeDataloader(AIFaceDataset("/home/brg2890/major/datasets/ai-face", ImageFileData, {}, AttributeNode, {"threshold": 5}))
trainer = DeepfakeAttributeTrainer(dataloader, .3, 1)