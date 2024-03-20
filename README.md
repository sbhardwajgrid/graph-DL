# Deep Learning for Graph Commmunity Detection
This module implements supervised Community Detection on the DBLP dataset ([link](https://snap.stanford.edu/data/com-DBLP.html))

## Setup:
1. Run setup.py or run command "pip install -r requirements.txt"

## Data
1. Download the raw data from the mentioned link and save it in "./data/raw" as "graph_edges.txt" and "5000_communities.txt"
2. Create 2 subfolders "./data/interim" and "./data/processed"
3. Raw data is processed by the dataloader script at "./src/data/data_loader.py". It currently supports 2 experiments "edge_sampling" and "n_communities".

## Explorations
1. Explore the dataset in "./notebooks/data_exploration.ipynb"

## Models:
1. Models architectures are defined in "./src/models/models.py"
2. Training functions for the different architectures are in "./src/models/training.py"

## Training:
1. Training loops are implemented in "./notebooks/model_building.ipynb"
2. Training logs are saved in "./src/logs"

## Results Evaluations
1. Training logs from "./src/logs" are analysed in "./notebooks/results_evaluation.ipynb"

## Pretrained models
1. Pretrained models are stored in "./models/"
2. These models are trianed on a subset of 200 communities with 50% edge sampling

## Testing
1. Testing is ongoing. Script available at "./tests/testing.ipynb"
2. For conducting your own tests, use the "./src/dataloader.py" to generate the data and use pretrained models to generate communities

## Reports and Presentation
1. Find the associated presentation, report and figures in "./src/reports/"
