# MLWaterHarvesting
This repository contains codes related to the publication "Investigating machine learning models to predict atmospheric water harvesting from an ion deposited membrane" ().
Datasets and trained models are published on our Zenodo repository https://doi.org/10.5281/zenodo.10533012.

In particular:

- Folder `HP tuning` contains all `.ipynb` files for finding the optimal hyper-parameter combinations through grid-search optimization. Specifically, each file will run a grid-search optimization over 50 different and mutually exclusive splits of training and testing set. The optimal combination is chosen according to a majority vote strategy.
- `Model_comparison.ipynb` trains the ML models and compares their accuracy and stability.
- `Simulation.ipynb` makes predictions over real world data.
