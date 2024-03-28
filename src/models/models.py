import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MLP(torch.nn.Module):

    """
        Basic Multi Layer Perceptron with "num_layers" layers, relu activation fn and sigmoid output layer
    """

    def __init__(self , input_dim , hidden_dim , output_dim , num_layers , dropout):

        super(MLP, self).__init__()

        self.dropout=dropout

        self.dense_layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, hidden_dim)])

        for i in range(num_layers-2):
            self.dense_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.dense_layers.append(torch.nn.Linear(hidden_dim , output_dim))

        self.output = torch.nn.Sigmoid()

    def reset_parameters(self):

        for layer in self.dense_layers:
            layer.reset_parameters()

    def forward(self, x):

        out = None
        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x , p=self.dropout , training=self.training)

        x = self.dense_layers[-1](x)
        x = torch.clip(x , -4 , 4)
        out = self.output(x)

        return out
    
class GCN(torch.nn.Module):

    """
        Simple GCN with "num_layers" layers
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(GCN, self).__init__()

        # Initialise GCNConv layers
        
        self.convs = torch.nn.ModuleList([GCNConv(input_dim , hidden_dim , add_self_loops=True , normalize=True)])
        for i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_dim , hidden_dim , add_self_loops=True , normalize=True))
        self.convs.append(GCNConv(hidden_dim , output_dim , add_self_loops=True , normalize=True))

        # Initialise batch normalization layers
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)])

        # Final output layer
        self.output = torch.nn.Sigmoid()

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x , edge_index):

        out = None

        for conv, batch_norm in zip(self.convs[:-1], self.bns):
            
            x = conv(x , edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x , edge_index)
        x = torch.clip(x , -4 , 4)
        
        if(self.return_embeds):
            out = x
        else:
            out = self.output(x)

        return out
    

class mlp_GCN(torch.nn.Module):

    """
        GCN implementation with a dense layer for pre and post processing to learn better features
    """

    def __init__(self , input_dim , encoding_dim , hidden_dim , output_dim , num_layers , dropout, return_embeds=False):

        super(mlp_GCN, self).__init__()

        # Initialise pre and post processing MLP layers
        self.mlp_preprocessor = torch.nn.ModuleList([torch.nn.Linear(input_dim, encoding_dim),
                                                    torch.nn.Linear(encoding_dim , encoding_dim)])
        
        self.mlp_postprocessor = torch.nn.ModuleList([torch.nn.Linear(hidden_dim , output_dim)])

        # Initialise GCNConv layers
        self.convs = torch.nn.ModuleList([GCNConv(encoding_dim , hidden_dim , add_self_loops=True , normalize=True)])
        for i in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim , hidden_dim , add_self_loops=True , normalize=True))

        # Initialise batch normalization layers
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)])

        # Final output layer
        self.output = torch.nn.Sigmoid()

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            bn.reset_parameters()

        for layer in self.mlp_preprocessor:
            layer.reset_parameters()

        for layer in self.mlp_postprocessor:
            layer.reset_parameters()


    def forward(self, x , edge_index):

        out = None

        for linear_layer in self.mlp_preprocessor:
            x = linear_layer(x)
            x = F.relu(x)

        for conv, batch_norm in zip(self.convs[:-1], self.bns):
            
            x = conv(x , edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x , edge_index)
        x = F.relu(x)
    
        for linear_layer in self.mlp_postprocessor[:-1]:
            x = linear_layer(x)
            x = F.relu(x)

        x = self.mlp_postprocessor[-1](x)
        x = F.tanh(x)

        if(self.return_embeds):
            out = x
        else:
            out = self.output(x)

        return out
    
class n2vnet(torch.nn.Module):

    """
        MLP on top of node2vec embeddings for classfication
    """

    def __init__(self , g , input_dim , hidden_dim , output_dim , num_layers , dropout):

        super(n2vnet, self).__init__()

        # Initialise the dense layers for classification task

        self.dense_layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, hidden_dim)])
        for i in range(num_layers-2):
            self.dense_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.dense_layers.append(torch.nn.Linear(hidden_dim , output_dim))

        self.output = torch.nn.Sigmoid()

        self.dropout = dropout

    def reset_parameters(self):

        for layer in self.dense_layers:
            layer.reset_parameters()

    def forward(self , embeddings):

        out = None

        # Forward pass is called on the embeddings generated by node2vec algorithm
        x = embeddings

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x , p=self.dropout , training=self.training)

        x = self.dense_layers[-1](x)
        out = self.output(x)

        return out
    
