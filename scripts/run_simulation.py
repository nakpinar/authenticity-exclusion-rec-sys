"""
Runs simulation with given parameters and saves the results as csvs
"""

import sys
sys.path.append("..")
import numpy as np
import argparse
from src.main_util import *

# Parse arguments
parser = argparse.ArgumentParser(description='Zipf')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--time_steps', type=int, default='10000')
parser.add_argument('--num_agents', type=int, default='1000')
parser.add_argument('--recommendation_policy', type=str, default='real_graph')
parser.add_argument('--group0_share', type=float, default=0.2)
parser.add_argument('--init_edges', type=str, default='homophilic')
parser.add_argument('--b0', type=float, default=1.)
parser.add_argument('--b1', type=float, default=1.)
parser.add_argument('--b2', type=float, default=5.)
parser.add_argument('--b3', type=float, default=-1.)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--p0', type=float, default=0.5)
parser.add_argument('--p1', type=float, default=0.4)
parser.add_argument('--p01', type=float, default=0.1)
parser.add_argument('--p10', type=float, default=0.1)
args = parser.parse_args()

# Set random seeds
np.random.seed(args.seed)

# Fixed arguments
preference_means=np.array([[5,1,4],[5,4,1]])
preference_cov=np.eye(3)/10
group_shares = np.array([args.group0_share,1-args.group0_share])

# Results path
b = [args.b0, args.b1, args.b2, args.b3]
b_str = '_'.join([str(x) for x in b])
p = [args.p0, args.p1, args.p01, args.p10]
p_str = '_'.join([str(x) for x in p])

results_path = os.path.abspath('../results/')
dir_path = os.path.join(results_path, f'policy_{args.recommendation_policy}_agents_{args.num_agents}_timesteps_{args.time_steps}_init_{args.init_edges}_groups_{str(group_shares)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seed_{args.seed}')

if not os.path.isdir(dir_path):
    os.makedirs(dir_path)

# Run simulation
t0 = time.time()
sim = Model(recommendation_policy=args.recommendation_policy,
            num_agents=args.num_agents,
            group_shares=group_shares,
            preference_means=preference_means,
            preference_cov=preference_cov,
            init_edges=args.init_edges,
            alpha=args.alpha,
            b=np.array(b),
            p=np.array(p),
            dir_path=dir_path,
            time_simulation=True)
sim.run(time_steps=args.time_steps)
t = time.time() - t0

# Save adjacency matrix
A = sim.adj_matrix
np.savetxt(dir_path + '/adj_matrix.csv', A, delimiter=",")
A = sim.smoothed_interaction_count[args.time_steps]
np.savetxt(dir_path + '/smoothed_interaction_count.csv', A, delimiter=",")
A = sim.get_edge_weights()
np.savetxt(dir_path + '/edge_weights.csv', A, delimiter=",")
groups = np.array([agent.group for agent in sim.agents])
np.savetxt(dir_path + '/agent_groups.csv', groups, delimiter=",")

print(f'Took: {t/60} minutes.')

# Save results
df = sim.get_res_agent()
df.to_csv(os.path.join(dir_path,'res_agent.csv'), index=False)
df = sim.get_res_content()
df.to_csv(os.path.join(dir_path,'res_content.csv'), index=False)
df = sim.get_res_recommendation()
df.to_csv(os.path.join(dir_path,'res_recommendation.csv'), index=False)
df = sim.get_res_timestep()
df.to_csv(os.path.join(dir_path,'res_timestep.csv'), index=False)
df = sim.get_res_topic()
df.to_csv(os.path.join(dir_path,'res_topic.csv'), index=False)
