import sys
sys.path.append('/Users/sbhardwaj/Documents/GraphNodeClassification')

from src.models.models import GCN

from fastapi import FastAPI , HTTPException
import torch
import networkx as nx
import numpy as np
from src.utilities.utils import NodeQueryResponse , CommunityQueryResponse , NewNodeQueryResponse , create_communities , create_nodeinfo , get_feature_vector 
from typing import List

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = "cpu"


app = FastAPI()

@app.post("/query_node/" , response_model=NodeQueryResponse)
async def get_node_info(node_id: int):

    """
    takes in node id and returns current groups and new groups sorted along with probability scores
    """

    data = torch.load(f"../../data/processed/data_16_200_0.5.pt")

    if node_id >= data.x.shape[0] or node_id<0:
        raise HTTPException(status_code=400, detail="Node does not exist")

    gcn_args = {
    'device': device,
    'num_layers': 4,
    'hidden_dim': 16,
    'dropout': 0.2,
}

    model_path = "../../models/gcn_16_200_0.5_1.pt"
    loaded_model = GCN(data.x.shape[1] , gcn_args['hidden_dim'] , data.y.shape[1] , gcn_args['num_layers'] , gcn_args['dropout'])
    loaded_model.load_state_dict(torch.load(model_path))

    model_output = loaded_model(data.x , data.edge_index)
    node_info = create_nodeinfo(model_output , thresh=0.9)

    actual_communities = list(int(x) for x in np.where(data.y[node_id].cpu() == 1)[0])

    new_communities = []
    for (comm , prob) in node_info[node_id]:
        if(comm in actual_communities):
            pass
        else:
            new_communities.append([comm , prob])

    return {"node_id":node_id , "output":node_info[node_id],
            "actual_communities":actual_communities , "new_communities":new_communities}

@app.post("/query_community/" , response_model=CommunityQueryResponse)
async def get_community_info(community_id: int):

    """
    takes community id as input and returns current members and new members sorted along with probability scores
    """

    data = torch.load(f"../../data/processed/data_16_200_0.5.pt")

    if community_id >= data.y.shape[1] or community_id<0:
        raise HTTPException(status_code=400, detail="Community does not exist")

    gcn_args = {
    'device': device,
    'num_layers': 4,
    'hidden_dim': 16,
    'dropout': 0.2,
    'lr': 0.005,
    'epochs': 1000}

    model_path = "../../models/gcn_16_200_0.5_1.pt"
    loaded_model = GCN(data.x.shape[1] , gcn_args['hidden_dim'] , data.y.shape[1] , gcn_args['num_layers'] , gcn_args['dropout'])
    loaded_model.load_state_dict(torch.load(model_path))

    model_output = loaded_model(data.x , data.edge_index)
    community_size = data.y[:,community_id].sum()
    communities = create_communities(model_output , thresh=0.97)

    actual_nodes = list(int(x) for x in np.where(data.y[: , community_id].cpu() == 1)[0])

    new_nodes = []

    for (node , prob) in communities[community_id]:
        if(node in actual_nodes):
            pass
        else:
            new_nodes.append([node , prob])
    
    return {"community_id":community_id , "community_size": community_size , "new_nodes_count":len(new_nodes),
            "output_size":len(communities[community_id]) , "actual_nodes":actual_nodes, "new_nodes":new_nodes[:100]}

@app.post("/query_new_node/" , response_model=NewNodeQueryResponse)
async def new_node(edge_list: List[int]):

    data = torch.load(f"../../data/processed/data_16_200_0.5.pt")

    if(len(edge_list) == 0):
        raise HTTPException(status_code=400, detail=f"Please enter non-empty list!")

    for node in edge_list:
        if node >= data.x.shape[0] or node<0:
            raise HTTPException(status_code=400, detail=f"Node {node} does not exist")

    gcn_args = {
    'device': device,
    'num_layers': 4,
    'hidden_dim': 16,
    'dropout': 0.2,
    'lr': 0.005,
    'epochs': 1000}

    model_path = "../../models/gcn_16_200_0.5_1.pt"
    loaded_model = GCN(data.x.shape[1] , gcn_args['hidden_dim'] , data.y.shape[1] , gcn_args['num_layers'] , gcn_args['dropout'])
    loaded_model.load_state_dict(torch.load(model_path))

    new_node_id = data.x.shape[0]
    new_node_feature_vector = get_feature_vector(new_node_id , edge_list , data.x.cpu() , data.g)

    # create edge_index (adjacency list)
    edge_index = np.array([[new_node_id , edge] for edge in edge_list])
    edge_index_ = np.flip(edge_index , axis=1)      # model expects directed edges so we make the edge list symmetrical
    self_loops = np.array([[new_node_id,new_node_id]])     # adding self loops
    edge_index = np.concatenate([data.edge_index.T.cpu() , edge_index , edge_index_ , self_loops] , axis=0)
    edge_index = torch.tensor(edge_index.T , dtype=torch.int32)

    x = np.concatenate([data.x.cpu() , new_node_feature_vector.reshape((1,-1))] , axis=0)
    x = torch.tensor(x , dtype=torch.float32)

    model_output = loaded_model(x , edge_index)
    node_info = create_nodeinfo(model_output , thresh=0.8)
    
    return {"new_node_id":new_node_id , "edge_list":edge_list , "recommended_communities":node_info[new_node_id]}