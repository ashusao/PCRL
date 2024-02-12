import pickle
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys, os
sys.path.append("..")  # Adds higher directory to python modules path.
import evaluation_framework as ef


def nodesize(station_list, my_graph, my_plan):
    ns = []
    for node in my_graph.nodes():
        if node not in station_list:
            ns.append(2)
        else:
            i = station_list.index(node)
            station = my_plan[i]
            try:
                capacity = station[2]["capability"]
            except KeyError:
                # exisiting charging infrastructure
                capacity = np.sum(ef.CHARGING_POWER * station[1])
            if capacity < 100:
                ns.append(6)
            elif 100 <= capacity < 200:
                ns.append(11)
            elif 200 <= capacity < 300:
                ns.append(16)
            else:
                ns.append(21)
    return ns


def visualise_stations(my_graph, my_plan, my_filepath):
    """
    Create plot of the charging station distribution
    """
    station_list = []
    for station in my_plan:
        station_list.append(station[0][0])
    colours = ['r', 'grey']
    nc = [colours[0] if node in station_list else colours[1] for node in my_graph.nodes()]
    labels = ['Charging station', 'Normal road junction']
    legend_elements = [Line2D([0], [0], marker='o', color='w', lw=0,
                              markerfacecolor=colours[0], markersize=7),
                       Line2D([0], [0], marker='o', color='w', lw=0,
                              markerfacecolor=colours[1], markersize=4)]
    ns = nodesize(station_list, my_graph, my_plan)
    fig, ax = ox.plot_graph(my_graph, node_color=nc, save=False, node_size=ns, edge_linewidth=0.2, edge_alpha=0.8,
                            show=False, close=False)
    ax.legend(legend_elements, labels, loc=2, prop={"size": 12})
    plt.savefig(my_filepath)
    plt.show()


if __name__ == '__main__':
    ox.config(use_cache=True, log_console=True)
    ort = "Toy_Example"

    G = ox.load_graphml("../Graph/" + ort + "/" + ort + ".graphml")

    with (open("../Results/" + ort + "/plan_RL.pkl", "rb")) as f:
        plan = pickle.load(f)

    os.makedirs("../Images/Result_Plots/" + ort, exist_ok=True)
    
    visualise_stations(G, plan, "../Images/Result_Plots/" + ort + "/RL_" + ort + ".png")

