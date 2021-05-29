# Predicting Restaurant Popularity in LBSNs with Graph Neural Networks
**Authors:** CHOI Sehyun, TILNEY-BASSETT Oktarian, LAU Yik Lun

## Overview
Location-based social networks (LSBNs) are widely popular and allow users to interact with businesses over the world. As such, the data collected by these networks offer opportunities for business owners to make more informed decisions using data-driven tools.
This project aims to create a comprehensive model by leveraging the power of LSBNs to predict the popularity of a business. For this project, we limit our scope to restaurants and food and drinks related businesses, since they are the majority business type stored in the target dataset, Yelp. This work is important as it could allow restaurant owners to better understand hidden factors that make restaurants successful.

## Source Code
The source code is fairly large as a wide range of experiments were conducted. The file organization is as follows:
- [Datasets](https://github.com/OktarianTB/predicting-business-popularity/tree/main/datasets): contains the pre-processed Yelp data
- [GNN](https://github.com/OktarianTB/predicting-business-popularity/tree/main/gnn): contain two different Graph Neural Network models (Restaurant GNN with GNN-Explainer, User-Restaurant GNN)
- [Graphs](https://github.com/OktarianTB/predicting-business-popularity/tree/main/graphs): contains some of the generated graphs used to train our models
- [GraphShop](https://github.com/OktarianTB/predicting-business-popularity/tree/main/graphshop): contains the code for the GraphShop model
- [Heterogeneous Graphs](https://github.com/OktarianTB/predicting-business-popularity/tree/main/heterogeneous_graphs): contains initial code for creating and managing heterogeneous graph, but this was not further explored
- [Popularity Metrics](https://github.com/OktarianTB/predicting-business-popularity/tree/main/popularity_metrics): contains all of our data analysis for selecting relevant popularity metrics, as well as other useful information regarding the datasets
- [Restaurant Data](https://github.com/OktarianTB/predicting-business-popularity/tree/main/restaurant_data): contains the notebooks used to extract edges between restaurants
- [Restaurant Graphs](https://github.com/OktarianTB/predicting-business-popularity/tree/main/restaurant_graphs): contains the notebook used to generate the Restaurant Graphs, including feature augmentation
- [Users Network](https://github.com/OktarianTB/predicting-business-popularity/tree/main/users_network): contains the notebook used to generate the User Graphs

## Datasets
The Yelp dataset can be downloaded directly from their [website](https://www.yelp.com/dataset).

Some of the other data files are too large to be uploaded to Github: they can be
found in the following [Onedrive link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/schoiaj_connect_ust_hk/ElDgqgTriKVMnkkrznvhfi8BselQ9tjS1yeys30kmO43iA?e=Eg4OY5).
