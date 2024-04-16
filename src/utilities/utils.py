import numpy as np
import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel
from typing import List
import networkx as nx

# To calculate accuracy,recall and precision given the confusion matrix
def acc(c):
    return ((c[0][0] + c[1][1])/(c.sum()))

def recall(c):
    return ((c[1][1])/(c[1][0] + c[1][1]))

def precision(c):
    return ((c[1][1])/(c[1][1] + c[0][1]))


# Fn to read and parse a log file. Returns a disctionary containing the relevant metrics
def read_log_file(log_file_path , metric="acc"):

    log_file = open(log_file_path , mode="r")

    epochs = []
    loss = []
    train = []
    val = []
    test = []


    for log in log_file:

        elements = log.split()

        epochs.append(int(elements[4]))
        loss.append(float(elements[6]))

        
        c_train = np.array([[float(elements[8]) , float(elements[9])],
                            [float(elements[10]) , float(elements[11])]])
        c_val = np.array([[float(elements[13]) , float(elements[14])],
                            [float(elements[15]) , float(elements[16])]])
        c_test = np.array([[float(elements[18]) , float(elements[19])],
                            [float(elements[20]) , float(elements[21])]])
                
        if(metric == "acc"):
            train.append(acc(c_train))
            val.append(acc(c_val))
            test.append(acc(c_test))

        elif(metric == "recall"):
            train.append(recall(c_train))
            val.append(recall(c_val))
            test.append(recall(c_test))

        elif(metric == "precision"):
            train.append(precision(c_train))
            val.append(precision(c_val))
            test.append(precision(c_test))

    log_file.close()

    return {
        "epochs":epochs,
        "loss":loss,
        "train":train,
        "val":val,
        "test":test
    }


# Make a plot by specifying the parameters
def plot_metrics(x , x_axis_label , y_list , y_labels , y_axis_label , colour_list , title , linewidth=1.5):

    plt.style.use('seaborn-v0_8-darkgrid')

    for i , y in enumerate(y_list):

        plt.plot(x , y , label=y_labels[i] , color=colour_list[i] , linestyle = '-' , linewidth=linewidth )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# Helper function for dataloader to sort dict values by the keys
def get_values_sorted_by_keys(d):

    keys = list(d.keys())
    keys.sort()
    sorted_vals = np.array([ d[key] for key in keys])

    return sorted_vals


# save model at specified path

def save_model(model , path):

    torch.save(model.state_dict(), path)


# Sorter 
def sorter(a):
    return a[1]

# used in test.ipynb
# converts output matrix to list of communities sorted by the probability of the node being a part of the community
    
def create_communities(out_matrix , thresh = 0.5):
    
    """
        input: Output of the model
        description: Converts o/p of the model to a usable format
        output: list of communities containing nodes sorted by their probability score (of being a part of the respective community)
    """

    out_ = out_matrix.to("cpu").detach().numpy()

    tuples = np.where(out_>thresh)
    scores = np.where(out_>thresh , out_ , -1)

    communities = [[] for i in range(out_.shape[1])]

    for i in range(tuples[0].shape[0]):

        node_id = tuples[0][i]
        community_id = tuples[1][i]

        communities[community_id].append(np.array([int(node_id) , scores[node_id][community_id]]))

    for i in range(len(communities)):
        communities[i].sort(key = sorter , reverse=True)

    return (communities)

def create_nodeinfo(out_matrix , thresh = 0.5):
    
    """
        input: Output of the model
        description: Converts o/p of the model to a usable format
        output: list[node_id] is a list of communities node_id is a part of
    """

    out_ = out_matrix.to("cpu").detach().numpy()

    tuples = np.where(out_>thresh)
    scores = np.where(out_>thresh , out_ , -1)

    nodeinfo = [[] for i in range(out_.shape[0])]

    for i in range(tuples[0].shape[0]):

        node_id = tuples[0][i]
        community_id = tuples[1][i]

        nodeinfo[node_id].append([int(community_id) , scores[node_id][community_id]])

    for i in range(len(nodeinfo)):
        nodeinfo[i].sort(key = sorter , reverse=True)

    return nodeinfo

# Response classes for apis

class NodeQueryResponse(BaseModel):
    node_id: int
    output: List[List[float]]
    actual_communities: List[int]
    new_communities: List[List[float]]

class CommunityQueryResponse(BaseModel):
    community_id: int
    community_size: int
    new_nodes_count: int
    output_size: int
    actual_nodes: List[int]
    new_nodes: List[List[float]]

class NewNodeQueryResponse(BaseModel):
    new_node_id: int
    edge_list: List[int]
    recommended_communities: List[List[float]]

def get_feature_vector(node_id , edge_list , x , g):
    "computes the feature vector for the new node based on its edge list and the graph structure"

    embedding = np.zeros((21,))
    for edge in edge_list:
        embedding += x[edge , :21].numpy()
    embedding = np.array(embedding)
    embedding/=len(edge_list)
    embedding = torch.tensor(embedding , dtype=torch.float32)

    # g.add_node(node_id)
    # edges = [(node_id , edge) for edge in edge_list]
    # g.add_edges_from(edges)

    # features = []
    # # degree, eigenvector centrality, clustering coefficient, square clustering coeff
    # features.append(g.degree([node_id]))
    # features.append(nx.eigenvector_centrality(g)[node_id])
    # features.append(nx.clustering(g)[node_id])
    # features.append(nx.square_clustering(g)[node_id])

    # features = torch.nn.functional.normalize(torch.tensor(features , dtype=torch.float32) , dim=0)
    # feature_vector = torch.cat([embedding , features] , dim=1)


    return embedding