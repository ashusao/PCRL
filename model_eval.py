from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import pickle, os
import env_plus as ev

"""
Generate a charging plan based on the model.
"""
# Instantiate the env
location = "Toy_Example"

graph_file = "Graph/" + location + "/" + location + ".graphml"
node_file = "Graph/" + location + "/nodes_extended_" + location + ".txt"
plan_file = "Graph/" + location + "/existingplan_" + location + ".pkl"

env = ev.StationPlacement(graph_file, node_file, plan_file)
log_dir = "tmp_Toy_Example/"

"""
Ab hier kommt die Evaluation.
"""
print("Evaluation for best model")
env = Monitor(env, log_dir)  # new environment for evaluation

# get the best model
prefix = "best_model_" + location + "_"
files = [f for f in os.listdir(log_dir) if f.startswith(prefix)]
modelname = prefix + str(max(int(f.split("_")[-1].split(".")[0]) for f in files)) + ".zip"
model = DQN.load(log_dir + modelname)

obs = env.reset()
done = False
best_plan, best_node_list = None, None
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        best_node_list, best_plan = env.render()

os.makedirs("Results/" + location, exist_ok=True)

pickle.dump(best_plan, open("Results/" + location + "/plan_RL.pkl", "wb"))
with open("Results/" + location + "/nodes_RL.txt", 'w') as file:
    file.write(str(best_node_list))
