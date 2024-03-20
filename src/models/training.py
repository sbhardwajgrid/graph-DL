import torch
import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix

class weighted_BCE(torch.nn.Module):

    def __init__(self , reduction="mean" , true_weight =  1.5 , false_weight = 0.5):

        super(weighted_BCE, self).__init__()
        self.reduction = reduction
        self.true_weight = true_weight
        self.false_weight = false_weight

    def forward(self, inputs, targets):

        loss = -1 * ((self.true_weight * targets * torch.log(inputs)) + (self.false_weight * (1 - targets) * torch.log(1 - inputs)))

        if(self.reduction == "sum"):
            return loss.sum()
        else:
            return loss.mean()

def train_gcn(model, data , optimizer , loss_fn):

    model.train()
    loss = 0

    optimizer.zero_grad()
    out = model(data.x , data.edge_index)

    y_hat = (out[data.train_mask])
    y = (data.y[data.train_mask])

    loss = loss_fn(y_hat , y)

    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test_gcn(model, data):
    
    model.eval()

    out = model( (data.x) , (data.edge_index))

    c_train = np.sum(multilabel_confusion_matrix((data.y[data.train_mask].int()).cpu() , ((out>0.5)[data.train_mask].int()).cpu()) , axis=0)
    c_val = np.sum(multilabel_confusion_matrix((data.y[data.val_mask].int()).cpu() , ((out>0.5)[data.val_mask].int()).cpu()) , axis=0)
    c_test = np.sum(multilabel_confusion_matrix((data.y[data.test_mask].int()).cpu() , ((out>0.5)[data.test_mask].int()).cpu()) , axis=0)

    return c_train , c_val , c_test

def train_mlp(model, data , optimizer , loss_fn):

    model.train()
    loss = 0

    optimizer.zero_grad()
    out = model(data.x)

    out_slice = (out[data.train_mask])
    out_labels = (data.y[data.train_mask])
    loss = loss_fn(out_slice , out_labels)

    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test_mlp(model, data):
    
    model.eval()


    out = model(data.x)

    c_train = np.sum(multilabel_confusion_matrix((data.y[data.train_mask].int()).cpu() , ((out>0.5)[data.train_mask].int()).cpu()) , axis=0)
    c_val = np.sum(multilabel_confusion_matrix((data.y[data.val_mask].int()).cpu() , ((out>0.5)[data.val_mask].int()).cpu()) , axis=0)    
    c_test = np.sum(multilabel_confusion_matrix((data.y[data.test_mask].int()).cpu() , ((out>0.5)[data.test_mask].int()).cpu()) , axis=0)

    return c_train , c_val , c_test

def train_n2v(model, embeddings , data , optimizer , loss_fn):

    model.train()
    loss = 0

    optimizer.zero_grad()
    out = model(embeddings)

    out_slice = (out[data.train_mask])
    out_labels = (data.y[data.train_mask])
    loss = loss_fn(out_slice , out_labels)

    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test_n2v(model, embeddings , data):
    
    model.eval()

    out = model(embeddings)

    c_train = np.sum(multilabel_confusion_matrix((data.y[data.train_mask].int()).cpu() , ((out>0.5)[data.train_mask].int()).cpu()) , axis=0)
    c_val = np.sum(multilabel_confusion_matrix((data.y[data.val_mask].int()).cpu() , ((out>0.5)[data.val_mask].int()).cpu()) , axis=0)    
    c_test = np.sum(multilabel_confusion_matrix((data.y[data.test_mask].int()).cpu() , ((out>0.5)[data.test_mask].int()).cpu()) , axis=0)

    return c_train , c_val , c_test




