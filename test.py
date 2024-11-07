import wandb
from trainers.Trainer import Trainer
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from datasets.AIFaceDataset import AIFaceDataset
from data.ImageFileData import ImageFileData
from nodes.atrnode import AttributeNode
from managers.NoGraphManager import NoGraphManager
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from traversals.IValueTraversal import IValueTraversal
from models.CNNModel import CNNModel
from edges.Edge import Edge

num_models = 1

dataloader = UnclusteredDeepfakeDataloader([AIFaceDataset("/home/brg2890/major/datasets/ai-face", ImageFileData, {}, AttributeNode, {"threshold": 2})], Edge)
graph = dataloader.load()
manager = NoGraphManager(graph)
train_traversal = IValueTraversal(manager.graph, num_models, 300, 10, .01)
test_traversal = ComprehensiveTraversal(manager.graph, num_models)
models = [CNNModel("/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/saved_models/test_" + str(i) + ".pt", "resnestdf") for i in range(num_models)]
trainer = Trainer(manager, train_traversal, test_traversal, models)
trainer.run()