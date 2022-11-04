# PCRL
# Reinforcement Learning-based Placement of Charging Stations in Urban Road Networks

<p align="center">
<img src="https://github.com/frantz03/PCRL/blob/main/RL_Dresden_mini.png" width="500">
</p>


# Description
This repository is the implementation of the project "Reinforcement Learning-based Placement of Charging Stations in Urban Road Networks" by Leonie von Wahl, Nicolas Tempelmeier, Ashutosh Sao and Elena Demidova. In this project, we train a model with Deep Q Network Reinforcement Learning to place charging stations in a road network.


# Installation
We use [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) as Reinforcement training framework. Moreover, we use [Gym](https://github.com/openai/gym) to create the RL environment and [OSMnx](https://github.com/gboeing/osmnx) to work with road networks. We use Python 3.8.10.

To install the requirements
```bash
git clone git@github.com:frantz03/PCRL.git
pip install -r final_requirements.txt
```

# Toy Example Dataset
Before training the models, some data are needed. The data preparation can be done with 
`load_graph.py` and `nodes_preparation.py`. However, we will not upload our own data here. 
Instead, we offer a preprocessed toy example of data. With this toy example, the training and 
evaluation can be tested.


# Training & Evaluation
To train a model on an example raod network run `reinforcement.py`. The custom environment
is described in `env_plus.py`.

To generate a charging plan from the model run `model_eval.py`.

Finally, to evaluate the charging plan with the metrics from the utility model ( in
`evaluation_framework.py`) run `test_solution.py`.

# Visualisation
To visulalise the results, run `visualise.py`.

# Folder structure 
For the data: `Graph/<location>/`

For the images: `Images/<location>/`

For the results: `Results/<location>/`



   

