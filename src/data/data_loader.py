import numpy as np
import pandas as pd
import networkx as nx
import torch
import os
import matplotlib.pyplot as plt
import node2vec

import sys
sys.path.append('/Users/sbhardwaj/Documents/GraphNodeClassification')
from src.utilities.utils import get_values_sorted_by_keys


from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import RandomNodeSplit


class DBLP_dataset(Dataset):

    """
    input: Raw data files at "root/raw"
    processing: creates frequencies.csv at "root/interim"
    output: creates processed dataset (data_0.pt) at "root/processed"
    """

    def __init__(self, root, raw_filenames , processed_filenames , expt = "edge_sampling" , edge_sampling = [1] , n_communities = [50 , 100 , 200 , 500] , transform=None, pre_transform=None):
        """
            root = Where the dataset should be stored. This folder is split
            into "raw" directory (downloaded dataset) and "processed" directory (processed data). 
        """
        self.raw_filenames = raw_filenames
        self.processed_filenames = processed_filenames
        self.edge_sampling = edge_sampling
        self.n_communities = n_communities
        self.root = root
        self.expt = expt

        super(DBLP_dataset , self).__init__(root, transform, pre_transform)

        self.data = None
        

    @property
    def raw_file_names(self):
        """ 
            If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  

            Filenames will be:
            "graph_edges.txt" and "5000_communities.txt"
        """
        return self.raw_filenames

    @property
    def processed_file_names(self):
        """ 
            If these files are found in "processed" directory, processing is skipped
        """
        
        return self.processed_filenames

    def download(self):
        """
            triggers download if "raw" does not contain specified raw_filenames
        """
        pass

    def process(self):

        """
            input: raw data files at "root/raw/*"
            description: processes raw datafiles for model use
            output: torch_geomteric.Data object saved as "data_0.pt" in "root/processed"
        """
        
        # node_list, edge_list, labels contain all nodes and edges
        node_list , edge_list = self.node_edge_list_from_txt_file(self.raw_paths[0])    # self.raw_paths[0] = "root/raw/self.raw_filenames[0]" = "root/raw/graph_edges.txt"

        h = nx.Graph()      # create nx graph for the dataset
        h.add_nodes_from(node_list)     # add all nodes
        h.add_edges_from(edge_list)     # add all edges

        f = self.get_frequencies()      # used for sampling value:count ("count" number of communities with "value" number of members)

        if(self.expt == "n_communities"):

            for idx , n_community in enumerate(self.n_communities):

                sampled_node_list = self.community_sampling(frequencies_file=f , k=n_community//20 , n_communities=n_community)        # sorted subset of nodes that are a part of the sampled communities
                embedding_ids = [str(node) for node in sampled_node_list]
                embs = torch.tensor(n2v_embeddings[embedding_ids] , dtype = torch.float32)

                # labels[i,j] = 1 if node i is a member of community j else 0
                labels = self.binary_encoded_communities_from_txt_file(os.path.abspath("..") + "/data/interim/modified_communities.txt" , node_list)     # self.raw_paths[1] = "root/raw/self.raw_filenames[1]" = "root/raw/5000_communities.txt"

                labels = torch.tensor(labels[sampled_node_list , :] , dtype=torch.float32)

                mapping = dict(zip(sampled_node_list , np.arange(0 , sampled_node_list.shape[0] , 1)))

                g = nx.Graph(h.subgraph(sampled_node_list).copy())      # subsample the selected nodes
                nx.relabel_nodes(g , mapping = mapping , copy=False)    # relabel nodes (due to model convention)

                edge_sampler = np.random.rand(1 , len(list(g.edges)))
                g.remove_edges_from((np.array(list(g.edges))[(edge_sampler<0.25)[0,:]]))
                
                features , g = self.get_feature_vector(g)     # get features for all nodes in subsampled graph
                features = torch.nn.functional.normalize(torch.tensor(feature_vector , dtype=torch.float32) , dim=0)
                feature_vector = torch.cat([embs , features] , dim=1)

                edge_index = np.array(g.edges)
                edge_index_ = np.flip(edge_index , axis=1)
                self_loops = np.array([[i,i] for i in g.nodes])
                edge_index = np.concatenate([edge_index , edge_index_ , self_loops] , axis=0)
                edge_index = torch.tensor(edge_index.T , dtype=torch.int32)
            
                data = Data(x=feature_vector , edge_index=edge_index , y=labels , dtype=torch.float32 , g=g)
                add_masks = RandomNodeSplit(split = "train_rest" , num_val=len(g.nodes)//10 , num_test=len(g.nodes)//10)
                data = add_masks(data)

                torch.save(data, os.path.join(self.processed_dir, f"data_nc_{idx}.pt") , pickle_protocol=4)

        elif(self.expt == "edge_sampling"):

            n_community = 200
            sampled_node_list = self.community_sampling(frequencies_file=f , k=n_community//20 , n_communities=n_community)        # subset of nodes that are a part of the sampled communities
            embedding_ids = [str(i) for i in range(sampled_node_list.shape[0])]

            # labels[i,j] = 1 if node i is a member of community j else 0
            labels = self.binary_encoded_communities_from_txt_file(os.path.abspath("..") + "/data/interim/modified_communities.txt" , node_list)     # self.raw_paths[1] = "root/raw/self.raw_filenames[1]" = "root/raw/5000_communities.txt"

            labels = torch.tensor(labels[sampled_node_list , :] , dtype=torch.float32)
            mapping = dict(zip(sampled_node_list , np.arange(0 , sampled_node_list.shape[0] , 1)))

            g = nx.Graph(h.subgraph(sampled_node_list).copy())      # subsample the selected nodes
            nx.relabel_nodes(g , mapping = mapping , copy=False)    # relabel nodes (due to model convention)
            n2v_embeddings = self.n2v_embeddings(g)      # get Node2Vec embeddings for all nodes
            embs = torch.tensor(n2v_embeddings[embedding_ids] , dtype = torch.float32)


            for idx , sample_threshold in enumerate(self.edge_sampling):
                
                edge_sampler = np.random.rand(1 , len(list(g.edges)))
                g.remove_edges_from((np.array(list(g.edges))[(edge_sampler<0.5)[0,:]]))
                
                features , g = self.get_feature_vector(g)     # get feature vector for all nodes in subsampled graph
                features = torch.nn.functional.normalize(torch.tensor(features , dtype=torch.float32) , dim=0)
                feature_vector = torch.cat([embs , features] , dim=1)

                edge_index = np.array(g.edges)
                edge_index_ = np.flip(edge_index , axis=1)
                self_loops = np.array([[i,i] for i in g.nodes])
                edge_index = np.concatenate([edge_index , edge_index_ , self_loops] , axis=0)
                edge_index = torch.tensor(edge_index.T , dtype=torch.int32)
            
                data = Data(x=feature_vector , edge_index=edge_index , y=labels , dtype=torch.float32 , g=g)
                add_masks = RandomNodeSplit(split = "train_rest" , num_val=len(g.nodes)//10 , num_test=len(g.nodes)//10)
                data = add_masks(data)

                torch.save(data, os.path.join(self.processed_dir, f"data_es_{idx}.pt") , pickle_protocol=4)



        # sampled_node_list = self.community_sampling(frequencies_file=f , k=5 , n_communities=100)        # subset of nodes that are a part of the sampled communities
        # # labels[i,j] = 1 if node i is a member of community j else 0
        # labels = self.binary_encoded_communities_from_txt_file(os.path.abspath("..") + "/data/interim/modified_communities.txt" , node_list)     # self.raw_paths[1] = "root/raw/self.raw_filenames[1]" = "root/raw/5000_communities.txt"

        # labels = torch.tensor(labels[sampled_node_list , :] , dtype=torch.float32)

        # mapping = dict(zip(sampled_node_list , np.arange(0 , sampled_node_list.shape[0] , 1)))

        # g = nx.Graph()      # create nx graph for the dataset
        # g.add_nodes_from(node_list)     # add all nodes
        # g.add_edges_from(edge_list)     # add all edges

        # g = nx.Graph(g.subgraph(sampled_node_list).copy())      # subsample the selected nodes
        # nx.relabel_nodes(g , mapping = mapping , copy=False)    # relabel nodes (due to model convention)
        
        # feature_vector = np.zeros((len(g.nodes) , num_feats))       # initialise the feature vector

        # feature_vector , g = self.get_feature_vector(g)     # get feature vector for all nodes in subsampled graph
        # feature_vector = torch.nn.functional.normalize(torch.tensor(feature_vector , dtype=torch.float32) , dim=0)

        # edge_index = np.array([e for e in g.edges])
        # edge_index_ = np.flip(edge_index , axis=1)
        # self_loops = np.array([[i,i] for i in g.nodes])
        # edge_index = np.concatenate([edge_index , edge_index_ , self_loops] , axis=0)
        # edge_index = torch.tensor(edge_index.T , dtype=torch.int32)
    
        # data = Data(x=feature_vector , edge_index=edge_index , y=labels , dtype=torch.float32 , g=g)
        # add_masks = RandomNodeSplit(split = "train_rest" , num_val=1000 , num_test=1000)
        # data = add_masks(data)
        # edge_sampler = np.random.rand(1 , data.edge_index.shape[1])

        # if (self.expt == "edge_sampling"):

        #     for idx , sample_threshold in enumerate(self.edge_sampling):

                
        #         data.edge_index = edge_index[: , (edge_sampler<sample_threshold)[0 , :]]
        #         torch.save(data, os.path.join(self.processed_dir, f"data_es_{idx}.pt") , pickle_protocol=4)

        # if(self.expt == "n_communities"):
        #     pass

            
    def node_edge_list_from_txt_file(self , file_path: str):

        """
            input: file_path(str), for "graph_edges.txt"
            description: creates sorted node_list and edge_list from "graph_edges.txt"
            output: (node_list , edge_list), 2 np arrays
        """

        txt_file = open(file_path)
        edge_list = []
        node_set = set([])

        for line in txt_file:    # each line in file contains an edge

            line_elements = line.split()    # split based on space

            if(line_elements[0] == "#"):    # ignore lines with metadata
                pass
            else:   
                edge = [int(x) for x in line_elements]      # edge in list form
                nodes = [int(x) for x in line_elements]     # nodes present in edge

                edge_list.append(edge)      # append edge
                node_set.update(nodes)      # update node set
        
        txt_file.close()
        
        edge_list = np.array(edge_list)
        node_list = np.sort(np.array(list(node_set)))       # convert set to list and return as sorted np array
        
        return node_list , edge_list
    
    
    def binary_encoded_communities_from_txt_file(self , file_path:str , node_list):
        
        """
            input: file_path(str): for "5000_communities.txt", 
                   node_list(np.array)
            description:  converts communities in .txt file to binary matrix format
            output: encoded_comms(np.array): shape=(max(node_list)+1 , total_communities)
        """

        txt_file = open(file_path)
        total_communities = 0

        for line in txt_file:       # each line in file is a community
            total_communities += 1

        encoded_comms = np.zeros((max(node_list)+1 , total_communities))

        txt_file.close()
        txt_file = open(file_path)

        community_id = 0

        for line in txt_file:
            line_elements = [int(x) for x in line.split()]      # nodes in community community_id
            encoded_comms[line_elements , community_id] = 1     # update matrix
            community_id+=1     # update community_idx


        txt_file.close()
        
        return encoded_comms
    
    

    def get_feature_vector(self , g):
        """
            input: g(Networkx graph)
            description: 
            takes in nx graph g, 
            calculates "n_features" stats for each node,
            updates the feature values in g and appends it to the feature vector
            output: (feature_vector: Pytorch tensor of shape (len(g.nodes) , n_features)) , g: nx graph with added node features)
        """

        feature_vector = np.ones((len(g.nodes) , 1))        # initialise the feature vector with a contant value of 1

        #Adding degree as a feature
        degree = (g.degree())     # compute node degrees
        nx.set_node_attributes(g , dict(degree) , "degree")     # Add the new features using nx.set_node_attributes(graph , feature: dict({node:value}) , name_of_feature: str)
        degree_ = get_values_sorted_by_keys(dict(degree))
        degree_ = np.reshape((degree_) , newshape=(-1,1))
        feature_vector = np.append(feature_vector , degree_ , axis=1)


        # Eigenvector Centrality
        e_centrality = nx.eigenvector_centrality(g , max_iter=500)     # compute Eigenvector centralities
        nx.set_node_attributes(g, e_centrality, "centrality")    # Add feature to graph
        e_centrality_ = get_values_sorted_by_keys(e_centrality)
        e_centrality_ = np.array((e_centrality_)).reshape((-1,1))        
        feature_vector = np.append(feature_vector , e_centrality_ , axis=1)     # Adding feature to feature vector

        # Clustering Coefficient
        cc = nx.clustering(g)    # compute clustering coefficients
        nx.set_node_attributes(g, cc, "clustering_coef")    # Add feature to graph
        cc_ = get_values_sorted_by_keys(cc)
        cc_ = np.array(cc_).reshape((-1,1))        
        feature_vector = np.append(feature_vector , cc_ , axis=1)     # Add feature to feature vector

        # Square clustering
        scc = nx.square_clustering(g)    # compute sqaure clustering coefficient
        nx.set_node_attributes(g, scc, "square_clustering_coef")    # Add feature to graph
        scc_ = get_values_sorted_by_keys(scc)
        scc_ = np.array(scc_).reshape((-1,1))        
        feature_vector = np.append(feature_vector , scc_ , axis=1)     # Add feature to feature vector

        # One hot encoded node id 
        # node_ids = np.array((g.nodes))      # compute node ids vector
        # node_ids_ohe = np.eye(len(node_ids))     # compute ohe
        # feature_vector = np.append(feature_vector , node_ids_ohe , axis=1)

        return feature_vector , g
            
    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if(self.expt=="edge_sampling"):
            data = torch.load(os.path.join(self.processed_dir, f'data_es_{idx}.pt'))
        elif(self.expt == "n_communities"):
            data = torch.load(os.path.join(self.processed_dir, f'data_nc_{idx}.pt'))
        return data
    
    def get_frequencies(self):

        """ 
            input: data_file_path(str): "5000_communities.txt" file
            description: counts the number of communities grouped by community member count
                         returns the pd.Dataframe and saves it to a .csv file
            output: frequencies(pd.Dataframe): columns:count,value
        """

        list_of_frequencies = []
        distribution_list = []

        data_file = open(self.raw_paths[1])

        for line in data_file:

            line_elements = [int(x) for x in line.split()]
            list_of_frequencies.append(len(line_elements))
            distribution_list.append(len(line_elements))


        unique, counts = np.unique(list_of_frequencies, return_counts=True)
        combined = np.stack([unique , counts] , axis=0)

        # creating a dataframe of frequencies to load.
        frequencies = pd.DataFrame(combined.T , columns=["value" , "count"])
        
        # Saving frequencies as csv file
        frequencies.to_csv(self.root + "/interim/frequencies.csv")

        data_file.close()
        # plt.hist(distribution_list , range=[0,101] , bins=100)
        # plt.show()

        return frequencies
    
    def community_sampling(self , frequencies_file , k = 5 , n_communities = 50):

        """
            input: frequencies_file(pd.Dataframe): value:count ("count" number of communities with "value" number of members)
                   data_file_path(str): file path for "5000_communities.txt" 
            description: selects communities from "5000_communities.txt" to form a balanced dataset of "n_communities" communities using frequencies_file
            output: modified_node_list(np.array): sorted list of nodes in the sampled communities.
                         
        """

        f = frequencies_file

        data_file = open(self.raw_paths[1])
        write_file = open(self.root + "/interim/modified_communities.txt", "w")

        k = k       # the average number of communties to be selected per frequency count

        modified_node_set = set([])
        distribution_list = []      # initialise list of member counts of selected communities
        count = 0       # variable to keep track of number of communities

        for line in data_file:

            line_elements = [int(x) for x in line.split()]

            random = np.random.rand()       # Generate a random number to do random sampling 
            # p_select is the probability of selecting community defined in the current line
            # p_select is directly proportional to k and inversely proportional to the number of communities with the same length 
            p_select = (k/(f[f["value"] == len(line_elements)]["count"].iloc[0]))

            if(count == n_communities):
                break

            elif(random <= p_select):       # select communities with probability p_select

                write_file.write(line)      # write the line(community) to "modified_communties.txt" file
                modified_node_set.update(line_elements)     # add nodes in community to the modified node set
                distribution_list.append(len(line_elements))        # add number of nodes in distribution_list to plot
                count+=1

            else:
                pass

        modified_node_list = np.sort(np.array(list(modified_node_set)))     # convert modified_node_set to np.array and sort

        data_file.close()
        write_file.close()

        return modified_node_list
    
    def n2v_embeddings(self , g , embedding_dim=16 , walk_length=10 , num_walks=100 , window=4 ,  min_count=3 , batch_words=5):

        """
            ip: graph
            desc: returns embeddings for each node 
            op: sorted node2vec embeddings for all nodes in g
        """

        sorted_node_list = (list(g.nodes))
        sorted_node_list.sort()
        sorted_node_list = [str(node) for node in sorted_node_list]

        embedding_args = {
            "embedding_dim":embedding_dim,
            "walk_length":walk_length,
            "num_walks":num_walks,
            "window":window,
            "min_count":min_count,
            "batch_words":batch_words
        }

        n2v = node2vec.Node2Vec(g , dimensions=embedding_args["embedding_dim"] , walk_length=embedding_args["walk_length"] , num_walks=embedding_args["num_walks"]) 
        embeddings = (n2v.fit(window=embedding_args["window"] , min_count=embedding_args["min_count"] , batch_words=embedding_args["batch_words"]).wv)

        return embeddings