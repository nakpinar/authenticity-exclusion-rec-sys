"""
Finding 3: Plot results over several runs, averaged
"""

import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import seaborn as sns

# Parse aguments
parser = argparse.ArgumentParser(description='Zipf')
parser.add_argument('--time_steps', type=int, default='10000')
parser.add_argument('--num_agents', type=int, default='1000')
parser.add_argument('--group0_share', type=float, default=0.2)
parser.add_argument('--b0', type=float, default=1.)
parser.add_argument('--b1', type=float, default=1.)
parser.add_argument('--b2', type=float, default=5.)
parser.add_argument('--b3', type=float, default=-1.)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--p0', type=float, default=0.5) #Stochastic block model parameters -- homophilic
parser.add_argument('--p1', type=float, default=0.4)
parser.add_argument('--p01', type=float, default=0.1)
parser.add_argument('--p10', type=float, default=0.1)
args = parser.parse_args()

np.random.seed(0)

# Fixed arguments
seeds = range(20)
colors = {'homophilic':'orange','complete':'green','random':'blue'}
preference_means = np.array([[5,1,4],[5,4,1]])
preference_cov = np.eye(3)/10
group_shares = np.array([args.group0_share,1-args.group0_share])

# Read results
b = [args.b0, args.b1, args.b2, args.b3]
b_str = '_'.join([str(x) for x in b])
p = [args.p0, args.p1, args.p01, args.p10]
p_str = '_'.join([str(x) for x in p])

# Create dir for plots
results_path = os.path.abspath('../results/')
plot_path = os.path.join(results_path, f'plots_agents_{args.num_agents}_timesteps_{args.time_steps}_groups_{str(group_shares)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seeds{min(seeds)}-{max(seeds)}')
if not os.path.isdir(plot_path):
    os.makedirs(plot_path)

results_path = os.path.abspath('../results/')
plot_path = os.path.join(results_path, f'plots_agents_{args.num_agents}_timesteps_{args.time_steps}_groups_{str(group_shares)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seeds{min(seeds)}-{max(seeds)}')
if not os.path.isdir(plot_path):
    os.makedirs(plot_path)

recommendation_policy = 'real_graph'
graph_type = 'homophilic'
results_path_policy = results_path + f'/{recommendation_policy}/'
df_share_content_all = pd.DataFrame()
df_share_rec_all = pd.DataFrame()

for seed in seeds:
    dir_path = os.path.join(results_path_policy,f'policy_{recommendation_policy}_agents_{args.num_agents}_timesteps_{args.time_steps}_init_{graph_type}_groups_{str(group_shares)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seed_{seed}')
    df = pd.read_csv(os.path.join(dir_path, 'res_recommendation.csv'), index_col=False)
    df_temp = df.copy()
    df_temp['agent_group'] = 3 # 3 for 'all'
    df = pd.concat([df, df_temp])
    df_share_rec = df.groupby(['agent_group','content_topic','time']).creator_group.value_counts(normalize=True).reset_index(name='share')

    # Add missing rows
    missing_rows = df_share_rec.loc[df_share_rec.share > 0.9999].copy().reset_index(drop=True)
    missing_rows['creator_group'] = 1 - missing_rows['creator_group']
    missing_rows['share'] = 0.0
    df_share_rec = pd.concat([df_share_rec, missing_rows])
    df_share_rec['seed'] = seed
    df_share_rec_all = pd.concat([df_share_rec_all,df_share_rec])

    # Shares among produced content
    df = pd.read_csv(os.path.join(dir_path, 'res_content.csv'), index_col=False)
    df_share_content = df.groupby(['topic','time']).creator_group.value_counts(normalize=True).reset_index(name='share')

    # Add missing rows
    missing_rows = df_share_content.loc[df_share_content.share > 0.9999].copy().reset_index(drop=True)
    missing_rows['creator_group'] = 1 - missing_rows['creator_group']
    missing_rows['share'] = 0.0
    df_share_content = pd.concat([df_share_content, missing_rows])
    df_share_content['seed'] = seed

    df_share_content_all = pd.concat([df_share_content_all,df_share_content])

# Check if filling in the missing 0% share columns were added correctly -- need to be the same number for both groups
df_share_rec_all.creator_group.value_counts()
df_share_content_all.creator_group.value_counts()

# Creation shares don't change over time (on average)
df_c = df_share_content_all
df_c = df_c.groupby(['topic','creator_group'])['share'].mean().reset_index()
df_c = df_c.loc[df_c.creator_group == 0]

# Box plot -- variation is time steps and simulation runs
df_r = df_share_rec_all[df_share_rec_all.creator_group == 0]
df_r = df_r.loc[df_r.time > 2500]
ax = plt.figure(figsize=(5,4))
df_r = df_r.loc[df_r.content_topic.isin([1,2])]
df_r['content_topic_str'] = df_r.content_topic.map({0: 'Professional \ntopic', 1: 'Mainstream \ntopic', 2: 'Marginal \ntopic'})
df_r['agent_group'] = df_r['agent_group'].map({0: 'To min. group', 1: 'To maj. group', 3: 'To all'})
sns.boxplot(x='content_topic_str', y='share', hue='agent_group', data=df_r, showmeans=True, meanprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'black','markersize':'5'})
plt.ylabel('Minority group rec. share')
plt.xlabel('')

for content_topic, share in zip(df_c.topic.values, df_c.share.values):
    if content_topic == 0:
        continue
    if content_topic == 1:
        plt.hlines(share, content_topic - 1.5, content_topic - 0.5, colors='red', linestyles='dashed', label='Min. content share')
    else:
        plt.hlines(share, content_topic - 1.5, content_topic - 0.5, colors='red', linestyles='dashed')

plt.grid(True)
legend = plt.legend(title='', loc='upper left')
legend.get_frame().set_alpha(0.5)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_box.png'))
plt.clf()

df_s = df_r.loc[df_r.agent_group == 'To all']
print(df_s.groupby(['content_topic','creator_group'])['share'].mean().reset_index())

df_s = df_r.loc[df_r.agent_group == 'To maj. group']
print(df_s.groupby(['content_topic','creator_group'])['share'].mean().reset_index())

df_s = df_r.loc[df_r.agent_group == 'To min. group']
print(df_s.groupby(['content_topic','creator_group'])['share'].mean().reset_index())