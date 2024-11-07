from trainers.Trainer import Trainer
import cv2
import torch

class IValueTrainer(Trainer):
    tags = ["i-value"]
    
    def __init__(self, graphmanager, train_traversal, test_traversal, models):
        super().__init__(graphmanager, train_traversal, test_traversal, models)
        
    
    def train(self):
        self.train_traversal.traverse()
        # For each model, add the current node to the batch
        for i in range(len(self.batches)):
            self.batches[i].append(self.train_traversal.get_pointers()[i]['current_node'])
        
        for i, [model, optim, loss] in enumerate(zip(self.models, self.optims, self.losses)):
            if len(self.batches[i]) == self.batch_size:
                model.train()
                batch = [self.transform(cv2.cvtColor(subdata.get_data().load_data(), cv2.COLOR_BGR2RGB)) for subdata in self.batches[i]]
                y_hat = model(torch.stack(batch).cuda())
                y = torch.tensor([subdata.get_label() for subdata in self.batches[i]]).unsqueeze(1).cuda()
                acc = self.train_acc.update(y_hat, y)
                self.train_acc_history.append(acc)
                # f1 = self.train_f1.update(y_hat, y)
                # auroc = self.train_auroc.update(y_hat, y)
                train_loss = loss(y_hat, y.float())
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                
                for subdata, prediction in zip(self.batches[i], y_hat):
                    self.stored_prediction_accuracy[i][subdata].append((prediction, subdata.get_label()))
                
                self.batches[i] = []
                
            elif len(self.batches[i]) > self.batch_size:
                raise ValueError("Batch size too large")
        
        return
