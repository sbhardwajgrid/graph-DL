from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="DBLP-Community-Detection-GCN",
    version="1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="S. Bhardwaj",
    description="GCN with Node2Vec for community detection on DBLP dataset",
    url="https://github.com/sbhardwajgrid/graph-DL"
)