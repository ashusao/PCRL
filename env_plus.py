import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
import pickle
from random import choice
from math import ceil
import itertools
import evaluation_framework as ef

"""
Custom environment
"""

def prepare_config():
    """
    we prepare the power capacities of the different charging configuration and to find the cheapest ones
    """
    N = len(ef.CHARGING_POWER)
    urn = list(range(0, ef.K + 1)) * N
    config_list = []
    for combination in itertools.combinations(urn, N):
        config_list.append(list(combination))

    my_config_dict = {}
    for config in config_list:
        if np.sum(config) > ef.K:
            continue
        else:
            capacity = np.sum(ef.CHARGING_POWER * config)
            if capacity in my_config_dict.keys():
                # check if we have found a better configuration for the same capacity
                if np.sum(ef.INSTALL_FEE * config) < np.sum(ef.INSTALL_FEE * my_config_dict[capacity]):
                    my_config_dict[capacity] = config
            else:
                my_config_dict[capacity] = config
    # if we have a cheaper price at more capacity, we will use that configuration even if less capacity is required
    key_list = sorted(list(my_config_dict.keys()))
    for index, key in enumerate(key_list):
        cost_list = [np.sum(ef.INSTALL_FEE * my_config_dict[my_key]) for my_key in key_list[index:]]
        best_cost_index = cost_list.index(min(cost_list)) + index
        best_config = my_config_dict[key_list[best_cost_index]]
        my_config_dict[key] = best_config
    return my_config_dict


def initial_solution(my_config_dict, my_node_list, s_pos):
    """
    get the initial solution for the charging configuration
    """
    W = 0  # minimum capacity constraint
    radius = 50
    for my_node in my_node_list:
        if ef.haversine(s_pos, my_node) <= radius:
            W += ef.weak_demand(my_node)
    W = ceil(W) * ef.BATTERY
    key_list = sorted(list(my_config_dict.keys()))
    for key in key_list:
        if key > W:
            break
    best_config = my_config_dict[key]
    return best_config


def coverage(my_node_list, my_plan):
    """
    see which nodes are covered by the charging plan
    """
    for my_node in my_node_list:
        cover = ef.node_coverage(my_plan, my_node)
        my_node[1]["covered"] = cover


def choose_node_new_benefit(free_list):
    """
    pick location which the smallest coverage
    """
    upbound_list = [my_node[1]["covered"] for my_node in free_list]
    pos_minindex = upbound_list.index(min(upbound_list))
    chosen_node = free_list[pos_minindex]
    return chosen_node


def choose_node_bydemand(free_list):
    """
    pick location with highest weakened demand
    """
    demand_list = [my_node[1]["demand"] * (1 - 0.1 * my_node[1]["private CS"]) for my_node in free_list]
    chosen_index = demand_list.index(max(demand_list))
    chosen_node = free_list[chosen_index]
    return chosen_node


def anti_choose_node_bybenefit(my_node_list, my_plan):
    """
    choose station with the least coverage
    """
    plan_list = [station[0][0] for station in my_plan]
    my_occupied_list = [node for node in my_node_list if node[0] in plan_list]
    if not my_occupied_list:
        return None
    upbound_list = [node[1]["upper bound"] for node in my_occupied_list]
    pos_minindex = upbound_list.index(min(upbound_list))
    remove_node = my_occupied_list[pos_minindex]
    plan_index = plan_list.index(remove_node[0])
    remove_station = my_plan[plan_index]
    return remove_station


def support_stations_hilfe(station):
    charg_time = station[2]["D_s"] / station[2]["service rate"]
    wait_time = station[2]["D_s"] * station[2]["W_s"]
    neediness = (wait_time + charg_time)
    return neediness


def support_stations(my_plan, free_list):
    """
    choose a station which needs support due to highest waiting + charging time
    """
    cost_list = [support_stations_hilfe(station) for station in my_plan]
    if not cost_list:
        chosen_node = choose_node_bydemand(free_list)
    else:
        index = cost_list.index(max(cost_list))
        station_sos = my_plan[index]
        if sum(station_sos[1]) < ef.K:
            chosen_node = station_sos[0]
        else:
            # look for nearest node that could support the station
            dis_list = [ef.haversine(station_sos[0], my_node) for my_node in free_list]
            min_index = dis_list.index(min(dis_list))
            chosen_node = free_list[min_index]
    return chosen_node


class Plan:
    def __init__(self, my_node_list, my_node_dict, my_cost_dict, my_plan_file):
        with (open(my_plan_file, "rb")) as f:
            self.plan = pickle.load(f)
        self.plan = [ef.s_dictionnary(my_station, my_node_list) for my_station in self.plan]
        my_node_list, _, _ = ef.station_seeking(self.plan, my_node_list, my_node_dict, my_cost_dict)
        # update the dictionnary
        self.plan = [ef.s_dictionnary(my_station, my_node_list) for my_station in self.plan]
        self.norm_benefit, self.norm_cost, self.norm_charg, self.norm_wait, self.norm_travel = \
            ef.existing_score(self.plan, my_node_list)
        self.existing_plan = self.plan.copy()
        self.existing_plan = [s[0] for s in self.existing_plan]

    def __repr__(self):
        return "The charging plan is {}".format(self.plan)

    def add_plan(self, my_station):
        self.plan.append(my_station)

    def remove_plan(self, my_station):
        self.plan.remove(my_station)

    def steal_column(self, stolen_station, my_budget):
        """
        steal a charger from the station, give budget back and check which charger type has been stolen
        """
        my_budget += stolen_station[2]["fee"]
        station_index = self.plan.index(stolen_station)
        # we choose the most expensive charging column
        if stolen_station[1][2] > 0:
            self.plan[station_index][1][2] -= 1
            config_index = 2
        elif stolen_station[1][1] > 0:
            self.plan[station_index][1][1] -= 1
            config_index = 1
        else:
            self.plan[station_index][1][0] -= 1
            config_index = 0
        if sum(stolen_station[1]) == 0:
            # this means we remove the entire stations as it only has one charger
            self.remove_plan(stolen_station)
        else:
            # the station remains, we only steal one charging column
            ef.installment_fee(stolen_station)
            my_budget -= stolen_station[2]["fee"]
        return my_budget, config_index


class Station:
    def __init__(self):
        self.s_pos = None
        self.s_x = None
        self.s_dict = {}
        self.station = [self.s_pos, self.s_x, self.s_dict]

    def __repr__(self):
        return "This station is {}".format(self.station)

    def add_position(self, my_node):
        self.station[0] = my_node

    def add_chargers(self, my_config):
        self.station[1] = my_config

    def establish_dictionnary(self, node_list):
        self.station = ef.s_dictionnary(self.station, node_list)


class StationPlacement(gym.Env):
    """Custom Environment that follows gym interface"""
    node_dict = {}
    cost_dict = {}

    def __init__(self, my_graph_file, my_node_file, my_plan_file):
        super(StationPlacement, self).__init__()
        _graph, self.node_list = ef.prepare_graph(my_graph_file, my_node_file)
        self.plan_file = my_plan_file
        self.node_list = [self.init_hilfe(my_node) for my_node in self.node_list]
        self.game_over = None
        self.budget = None
        self.plan_instance = None
        self.plan_length = None
        self.row_length = 5
        self.best_score = None
        self.best_plan = None
        self.best_node_list = None
        self.schritt = None
        self.config_dict = None
        # new action space including all charger types
        self.action_space = spaces.Discrete(5)
        shape = (self.row_length + len(ef.CHARGING_POWER)) * len(self.node_list) + 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(shape,), dtype=np.float16)

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        self.budget = ef.BUDGET
        self.game_over = False
        self.plan_instance = Plan(self.node_list, StationPlacement.node_dict, StationPlacement.cost_dict,
                                  self.plan_file)
        self.best_score, _, _, _, _, _ = ef.norm_score(self.plan_instance.plan, self.node_list,
                                                       self.plan_instance.norm_benefit, self.plan_instance.norm_charg,
                                                       self.plan_instance.norm_wait, self.plan_instance.norm_travel)
        self.plan_length = len(self.plan_instance.existing_plan)
        self.schritt = 0
        self.best_plan = []
        self.best_node_list = []
        self.config_dict = prepare_config()
        coverage(self.node_list, self.plan_instance.plan)
        obs = self.establish_observation()
        return obs

    def init_hilfe(self, my_node):
        StationPlacement.node_dict[my_node[0]] = {}  # prepare node_dict
        StationPlacement.cost_dict[my_node[0]] = {}
        my_node[1]["charging station"] = None
        my_node[1]["distance"] = None
        return my_node

    def establish_observation(self):
        """
        Build observation matrix
        """
        row_length = self.row_length + len(ef.CHARGING_POWER)
        width = row_length * len(self.node_list) + 1
        obs = np.zeros((width,))
        for j, node in enumerate(self.node_list):
            i = j * row_length
            obs[i + 0] = node[1]['x']
            obs[i + 1] = node[1]['y']
            obs[i + 2] = node[1]['demand']
            obs[i + 3] = node[1]['estate price']
            obs[i + 4] = node[1]['private CS']
            for my_station in self.plan_instance.plan:
                if my_station[0][0] == node[0]:
                    for e in range(len(ef.CHARGING_POWER)):
                        index = 5 + e
                        obs[i + index] = my_station[1][e]
                    break
        obs[-1] = self.budget
        obs = np.divide(obs, ef.BUDGET)
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        return obs

    def budget_adjustment(self, my_station):
        inst_cost = my_station[2]["fee"]
        if self.budget - inst_cost > 0:
            # if we have enough money, we build the station
            self.budget -= inst_cost
        else:
            self.game_over = True

    def budget_adjustment_small(self, config_index):
        if self.budget - ef.INSTALL_FEE[config_index] > 0:
            # if we have enough money, we build the charger
            self.budget -= ef.INSTALL_FEE[config_index]
        else:
            self.game_over = True

    def prepare_score(self):
        """
        We have to make a loop to reorganise the station assignment
        """
        for j in range(2):
            self.node_list, _, _ = ef.station_seeking(self.plan_instance.plan, self.node_list, StationPlacement.node_dict,
                                             StationPlacement.cost_dict)
            for i in range(len(self.plan_instance.plan)):
                self.plan_instance.plan[i] = ef.total_number_EVs(self.plan_instance.plan[i], self.node_list)
                self.plan_instance.plan[i] = ef.W_s(self.plan_instance.plan[i])
            j += 1

    def step(self, my_action):
        """
        Perform a step in the episode
        """
        chosen_node, free_list_zero, config_index, action = self._control_action(my_action)
        if chosen_node in free_list_zero:
            # build new station
            default_config = initial_solution(self.config_dict, self.node_list, chosen_node)
            station_instance = Station()
            station_instance.add_position(chosen_node)
            station_instance.add_chargers(default_config)
            station_instance.establish_dictionnary(self.node_list)
            # Step: Control budget
            self.budget_adjustment(station_instance.station)
            if not self.game_over:
                self.plan_instance.add_plan(station_instance.station)
        else:
            # add column to existing CS
            station_index = None
            for station in self.plan_instance.plan:
                if station[0][0] == chosen_node[0]:
                    station_index = self.plan_instance.plan.index(station)
                    break
            # Step: Control budget
            self.budget_adjustment_small(config_index)
            if not self.game_over:
                self.plan_instance.plan[station_index][1][config_index] += 1
        # Step: calculate reward
        reward = self.evaluation()
        coverage(self.node_list, self.plan_instance.plan)
        obs = self.establish_observation()
        # episode end conditions
        if len(self.plan_instance.plan) == len(self.node_list):
            self.game_over = True
        self.schritt += 1
        if self.schritt >= len(self.node_list) / 2:
            self.game_over = True
        # if self.game_over:
        #     print("Best score {}.".format(self.best_score))
        return obs, reward, self.game_over, {}

    def station_config_check(self, my_station):
        """
        no more than K chargers are allowed at the station
        """
        capacity = True
        if sum(my_station[1]) >= ef.K:
            capacity = False
        return capacity

    def _control_action(self, chosen_action):
        """
        we have three possibilities here: either build a new station, add a charger to an exisiting station or move a
        charger from an exisiting station to a station in need
        """
        my_action = chosen_action
        config_index = None
        full_station_list = [s[0][0] for s in self.plan_instance.plan if self.station_config_check(s)
                             is False]  # these are the stations with exactly K chargers
        station_list = [s[0][0] for s in self.plan_instance.plan]  # all charging stations
        occupied_list = [node for node in self.node_list if node[0] not in full_station_list and node[0] in
                         station_list]  # nodes with non-full stations
        free_list = [node for node in self.node_list if node[0] not in station_list]  # nodes without stations
        if 0 <= my_action <= 1:
            # build
            if my_action == 0:
                chosen_node = choose_node_new_benefit(free_list)
            else:
                chosen_node = choose_node_bydemand(free_list)
        elif 2 <= my_action <= 3:
            # add column to existing station
            config_index = 1
            if len(occupied_list) == 0:
                chosen_node = choice(free_list)
            else:
                if my_action == 2:
                    chosen_node = choose_node_new_benefit(occupied_list)
                else:
                    chosen_node = choose_node_bydemand(occupied_list)
        else:
            # move station
            steal_plan = [s for s in self.plan_instance.plan if s[0] not in self.plan_instance.existing_plan]
            # we can not steal from the existing charging plan
            stolen_station = anti_choose_node_bybenefit(self.node_list, steal_plan)
            if stolen_station is None:
                # only necessary if we take this action in the very beginning
                chosen_node = choice(free_list)
            else:
                self.budget, config_index = self.plan_instance.steal_column(stolen_station, self.budget)
                chosen_node = support_stations(self.plan_instance.plan, free_list)
        return chosen_node, free_list, config_index, my_action

    def evaluation(self):
        """
        Calculate the reward
        """
        reward = 0
        self.prepare_score()
        new_score, _, _, _, _, _ = ef.norm_score(self.plan_instance.plan, self.node_list,
                                                 self.plan_instance.norm_benefit, self.plan_instance.norm_charg,
                                                 self.plan_instance.norm_wait, self.plan_instance.norm_travel)
        new_score = max(new_score, -25)  # if negative score
        if new_score - self.best_score > 0:
            reward += (new_score - self.best_score)
            # avoid jojo learning
            self.best_score = new_score
            self.best_plan = self.plan_instance.plan.copy()
            self.best_node_list = self.node_list.copy()
        return reward

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        """
        print(f'Score is: {self.best_score}')
        print(f'Number of stations in charging plan: {len(self.plan_instance.plan)}')
        return self.best_node_list, self.best_plan


if __name__ == '__main__':
    location = "Toy_Example"
    graph_file = "Graph/" + location + "/" + location + ".graphml"
    node_file = "Graph/" + location + "/nodes_extended_" + location + ".txt"
    plan_file = "Graph/" + location + "/existingplan_" + location + ".pkl"
    env = StationPlacement(graph_file, node_file, plan_file)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
