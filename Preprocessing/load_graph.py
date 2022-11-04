import osmnx as ox
from shapely.geometry import Point, MultiPolygon, LineString, Polygon
from shapely.ops import split
import numpy as np
import pickle

"""
create graph and grid for the test set Hannover
"""
n_x, n_y = 32, 32
# coordinates for Hannover
rec = [(9.6044300, 52.3051373), (9.6044300, 52.4543349), (9.9184259, 52.4543349), (9.9184259, 52.3051373)]


def grid_preparation(my_rec):
    """
    put a grid over a rectangle
    :param my_rec: list of four GPS coordinates
    """
    result = Polygon(my_rec)
    # compute splitter
    minx, miny, maxx, maxy = result.bounds
    dx = (maxx - minx) / n_x  # width of a small part
    dy = (maxy - miny) / n_y  # height of a small part
    horizontal_splitters = [LineString([(minx, miny + k * dy), (maxx, miny + k * dy)]) for k in range(n_y)]
    vertical_splitters = [LineString([(minx + k * dx, miny), (minx + k * dx, maxy)]) for k in range(n_x)]
    splitters = horizontal_splitters + vertical_splitters
    # split
    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))
    my_grids = list(result.geoms)
    return my_grids


def grid_ret_index(point, grids):
    for j in range(len(grids)):
        if point.within(grids[j]):
            r, c = (n_x - int(j / n_x)) - 1, int(j % n_x)
            return r, c
    return None, None


def grid_location(my_node, grids):
    # finding in which cell of the grid the node is
    lon, lat = my_node[1]['x'], my_node[1]['y']
    point = Point(lon, lat)
    row, column = grid_ret_index(point, grids)
    my_node[1]["row"], my_node[1]["column"] = row, column


if __name__ == '__main__':
    G = ox.graph_from_address('350 5th Ave, New York, New York', network_type='drive', dist=250)

    # calculate for each node in which cell it is and for each grid cell how many nodes it has inside
    node_list = list(G.nodes(data=True))
    print(len(node_list))
    grids = grid_preparation(rec)
    grid_density = np.zeros((n_x, n_y))
    for node in node_list:
        grid_location(node, grids)
        row, column = node[1]["row"], node[1]["column"]
        grid_density[row][column] += 1
    print("Grid density is calculated.")

    # Save the graph files
    location = "Toy_Example"
    ox.save_graphml(G, filepath="../Graph/" + location + "/" + location + ".graphml")
    with open("../Graph/" + location + "/node_list_" + location + ".txt", 'w') as file:
        file.write(str(node_list))
    pickle.dump(grid_density, open("../Graph/" + location + "/grid_density_" + location + ".pkl", "wb"))
