import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import evaluation_framework as ef


"""
Prepare the nodes of the road network.
"""
n_x, n_y = 32, 32


def social_efficiency_upper_bound(my_node, my_node_list):
    """
    calculate the social efficiency for each node
    """
    priv_CS = my_node[1]["private CS"]
    I1_max = 0  # dimensionless
    for other_node in my_node_list:
        # calculate distance with haversine approximation
        if ef.haversine(my_node, other_node) <= ef.RADIUS_MAX:
            I1_max += 1
    my_node[1]["I1_max"] = I1_max
    delta_benefit = I1_max * (1 - 0.1 * priv_CS)
    delta_benefit /= 100  # does not matter as we scale here
    upper_bound = ef.my_lambda * delta_benefit / my_node[1]["estate price"]
    return upper_bound


def charging_demand(my_node):
    """
    calculates the charging demand for each node
    """
    if my_node[1]["row"] is None:
        my_node[1]["demand"] = np.mean(demand_matrix)
    else:
        # demand per unit time interval, therefore dimensionless
        my_node[1]["demand"] = demand_matrix[my_node[1]["row"]][my_node[1]["column"]]


def modify_demand_dict(my_demand_matrix):
    """
    adapt demand to the number of nodes in each grid cell
    """
    for i in range(n_x):
        for j in range(n_y):
            if grid_density[i][j] > 0:
                my_demand_matrix[i][j] = my_demand_matrix[i][j] / grid_density[i][j]
            else:
                my_demand_matrix[i][j] = 0
            my_demand_matrix[i][j] += demand_min
            if my_demand_matrix[i][j] >= 250:
                my_demand_matrix[i][j] = 250
    demand_max = np.amax(my_demand_matrix)
    my_demand_matrix /= demand_max
    return my_demand_matrix


if __name__ == '__main__':
    location = "Toy_Example"
    graph_file = "../Graph/" + location + "/" + location + ".graphml"
    node_file = "../Graph/" + location + "/node_list_" + location + ".txt"
    graph, node_list = ef.prepare_graph(graph_file, node_file)

    # open Pickle files with necessary data (in grid structure)
    with (open("../Graph/" + location + "/demand_" + location + ".pkl", "rb")) as f:
        objects = pickle.load(f)

    with (open("../Graph/" + location + "/grid_density_" + location + ".pkl", "rb")) as f:
        grid_density = pickle.load(f)

    with (open("../Graph/" + location + "/privCS_" + location + ".pkl", "rb")) as f:
        priv_matrix = pickle.load(f)

    with (open("../Graph/" + location + "/estateprice_" + location + ".pkl", "rb")) as f:
        estate_matrix = pickle.load(f)

    """
    Preparation of the nodes. 1) Demand 
    """
    demand_min = 0.05
    demand_matrix = objects
    demand_matrix = modify_demand_dict(demand_matrix)

    for node in node_list:
        charging_demand(node)

    """
    2.) Estate price
    """
    for node in node_list:
        if node[1]["row"] is None:
            node[1]["estate price"] = np.mean(estate_matrix)
        else:
            node[1]["estate price"] = estate_matrix[node[1]["row"]][node[1]["column"]]

    """
    3.) Private Charging stations
    """
    for node in node_list:
        if node[1]["row"] is None:
            node[1]["private CS"] = np.mean(priv_matrix)
        else:
            node[1]["private CS"] = priv_matrix[node[1]["row"]][node[1]["column"]]

    """
    4.)  Maximum of nodes covered if this node becomes a charging station.
    """
    for node in node_list:
        node[1]["upper bound"] = social_efficiency_upper_bound(node, node_list)

    """
    5.) Existing charging infrastructure
    """
    # open CSV File with the existing charging stations in it
    df = pd.read_csv("../data/" + location + "/OCM_simple" + location + ".csv")
    existing_plan = []
    for row in df.iterrows():
        # each charging station has assigned its nearest OSM node
        found_node = row[1]['nearest node']
        # number of chargers at each charging station
        number = row[1]['numberofpoints']
        node_index = [my_node[0] for my_node in node_list].index(found_node)
        s_pos = node_list[node_index]
        if s_pos in [s[0] for s in existing_plan]:
            continue
        else:
            s_x = np.array([0, number, 0])
            existing_plan.append([s_pos, s_x, {}])

    # Save node files
    with open("../Graph/" + location + "/nodes_extended_" + location + ".txt", 'w') as file:
        file.write(str(node_list))
    pickle.dump(existing_plan, open("../Graph/" + location + "/existingplan_" + location + ".pkl", "wb"))
