# Deep Learning for Graph Commmunity Detection
This module implements supervised Community Detection on the DBLP dataset ([link](https://snap.stanford.edu/data/com-DBLP.html))

## APIs and XAI 
1. Code for XAI is at "./notebooks/xai.ipynb"
2. Code for APIs is at "./src/api/main.py"
    run command "uvicorn main:app --port 8000" from directory "./src/api" to test apis using swagger ui for fast api
3. Report for project 3: "./reports/APIs_XAI_report.pdf"

## Setup:
1. Run setup.py or run command "pip install -r requirements.txt"

## Data
1. Download the raw data from the mentioned link and save it in "./data/raw" as "graph_edges.txt" and "5000_communities.txt"
2. Create 2 subfolders "./data/interim" and "./data/processed"
3. Raw data is processed by the dataloader script at "./src/data/data_loader.py".
    This script takes the raw data files "graph_edges.txt" and "5000_communities.txt" in "./data/raw/" as inputs and processes them for the model to use for training and testing. It saves the in processed data at "./data/processed/". The name of the processed file needs to be passed to the dataloader as the argument processed_filename. The file will be saved as "./data/processed/processed_filename". Make sure to use ".pt" extension for the processed filename.

    The processed_filenames currently saved are in the format "data_{n2v_embedding_size}_{number_of_output_communties}_{fraction_of_sampled_edges}.pt"

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
