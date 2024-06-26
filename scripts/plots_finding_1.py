"""
Finding 1: Plot results over several runs, averaged
"""

import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse


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
recommendation_policies = ['real_graph','random','topic']
init_edges = ['homophilic','complete','random']
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

# Number of times content of type 0 was recommended by group
for recommendation_policy in recommendation_policies:
    results_path_policy = results_path + f'/{recommendation_policy}/'
    plot_df = pd.DataFrame()

    for graph_type in init_edges:
        df_all = pd.DataFrame()

        for seed in seeds:
            dir_path = os.path.join(results_path_policy, f'policy_{recommendation_policy}_agents_{args.num_agents}_timesteps_{args.time_steps}_init_{graph_type}_groups_{str(group_shares)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seed_{seed}')
            df_content = pd.read_csv(os.path.join(dir_path,'res_content.csv'),index_col=False)
            df = df_content
            df['seed'] = seed
            df_all = pd.concat([df_all,df])

        df = df_all
        df_res = df.groupby(['topic','creator_group'])['count'].mean().reset_index()
        df = df.groupby(['topic', 'creator_group', 'time'])['count'].mean().reset_index()

        # Get ratios
        df0 = df.loc[df.creator_group == 0].sort_values(by=['topic','creator_group','time']).reset_index(drop=True)
        df0.rename(columns={'count':'count0'}, inplace=True)
        df0 = df0.drop(columns=['creator_group'])
        df1 = df.loc[df.creator_group == 1].sort_values(by=['topic','creator_group','time']).reset_index(drop=True)
        df1.rename(columns={'count': 'count1'}, inplace=True)
        df1 = df1.drop(columns=['creator_group'])

        df = pd.merge(df0, df1, on = ['topic','time'], how='outer')
        if df.isnull().any().any():
            raise ValueError
        df['ratio'] = df['count0'] / (df['count1'])
        df['graph_type'] = graph_type
        plot_df = pd.concat([plot_df, df])


    wide_df = plot_df
    window_size = 1000
    wide_df = wide_df.sort_values(by=['graph_type', 'time'])
    wide_df['mean_mavg'] = wide_df.groupby('graph_type')['ratio'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean())
    wide_df = wide_df.loc[wide_df.time > 2500]
    wide_df = wide_df.loc[wide_df.topic == 0]

    # Ratio of average per post recommendations for topic 0 posts, MAVG, result averaged over n runs
    plt.figure(figsize=(5, 4))
    for graph_type in wide_df.graph_type.unique():
        plt.plot(wide_df.loc[wide_df.graph_type == graph_type, 'time'], wide_df.loc[wide_df.graph_type == graph_type, 'mean_mavg'], color=colors[graph_type], label=graph_type)
    if recommendation_policy != 'real_graph':
        plt.ylim(0.7, 1.1)
    plt.xlabel('Time')
    plt.ylabel('Rec. minority group / Rec. majority group')
    # plt.title('Moving Average of Mean Value over Time for Different Groups')
    plt.legend(title='Network structure')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(plot_path,f'{recommendation_policy}_ratio_rec_0.png'))
    plt.clf()

# Same but for different group sizes
print('Different group sizes...')
recommendation_policy = 'real_graph'
results_path_policy = results_path + f'/{recommendation_policy}_size/'
plot_df = pd.DataFrame()
graph_type = 'homophilic'

for min_share in [0.1,0.2,0.3,0.4,0.5]:
    group_shares_temp = np.array([min_share, 1 - min_share])
    df_all = pd.DataFrame()

    for seed in seeds:
        dir_path = os.path.join(results_path_policy, f'policy_{recommendation_policy}_agents_{args.num_agents}_timesteps_{args.time_steps}_init_{graph_type}_groups_{str(group_shares_temp)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seed_{seed}')
        df_content = pd.read_csv(os.path.join(dir_path,'res_content.csv'),index_col=False)
        df = df_content
        df['seed'] = seed
        df_all = pd.concat([df_all,df])

    df = df_all

    # Get ratios
    df = df.groupby(['topic', 'creator_group', 'time', 'seed'])['count'].mean().reset_index()
    df0 = df.loc[df.creator_group == 0].sort_values(by=['topic','creator_group','time','seed']).reset_index(drop=True)
    df0.rename(columns={'count':'count0'}, inplace=True)
    df0 = df0.drop(columns=['creator_group'])
    df1 = df.loc[df.creator_group == 1].sort_values(by=['topic','creator_group','time','seed']).reset_index(drop=True)
    df1.rename(columns={'count': 'count1'}, inplace=True)
    df1 = df1.drop(columns=['creator_group'])

    df = pd.merge(df0, df1, on = ['topic','time','seed'], how='outer')
    df['count0'] = df['count0'].fillna(0)
    df['count1'] = df['count1'].fillna(0)
    df['ratio'] = df['count0'] / df['count1']

    wide_df = df.groupby(['topic','time'])['ratio'].describe().reset_index()
    wide_df['high'] = wide_df['mean'] + wide_df['std']
    wide_df['low'] = wide_df['mean'] - wide_df['std']
    wide_df['graph_type'] = graph_type
    wide_df['min_share'] = min_share

    plot_df = pd.concat([plot_df,wide_df])

wide_df = plot_df
window_size = 1000
wide_df = wide_df.sort_values(by=['graph_type', 'min_share', 'time'])
wide_df['mean_mavg'] = wide_df.groupby(['graph_type','min_share'])['mean'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean())
wide_df = wide_df.loc[wide_df.time > 2500]

wide_df = wide_df.loc[wide_df.topic == 0]

colors_share = {0.1: '#87CEFA',  # Light Sky Blue
                0.2: '#1E90FF',  # Dodger Blue
                0.3: '#4169E1',  # Royal Blue
                0.4: '#0000CD',  # Medium Dark Blue
                0.5: '#00008B'   # Dark Blue
                }

# Ratio of average per post recommendations for topic 0 posts, MAVG, result averaged over n runs
plt.figure(figsize=(5,4))
for min_share in wide_df.min_share.unique():
    plt.plot(wide_df.loc[(wide_df.graph_type == graph_type) & (wide_df.min_share == min_share),'time'], wide_df.loc[(wide_df.graph_type == graph_type) & (wide_df.min_share == min_share),'mean_mavg'], color=colors_share[min_share], label=min_share)
plt.xlabel('Time')
plt.ylabel('Rec. minority group / Rec. majority group')
legend = plt.legend(title='Minority \ngroup \nshare', loc='lower right')
legend.get_frame().set_alpha(0.5)
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(plot_path, f'size_{recommendation_policy}_ratio_rec_0.png'))
plt.clf()


# Interaction count among served topic 0 recommendations grouped by in-out-group over time
results_path = os.path.abspath('../results/')
recommendation_policy = 'real_graph'
results_path_policy = results_path + f'/{recommendation_policy}/'
plt.figure(figsize=(5, 4))
for graph_type in init_edges:
    df_all = pd.DataFrame()

    for seed in seeds:
        dir_path = os.path.join(results_path_policy,f'policy_{recommendation_policy}_agents_{args.num_agents}_timesteps_{args.time_steps}_init_{graph_type}_groups_{str(group_shares)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seed_{seed}')
        df = pd.read_csv(os.path.join(dir_path, 'res_recommendation.csv'), index_col=False)
        df = df.groupby(['time','agent_group','creator_group','content_topic']).interaction_count.mean().reset_index()
        df = df.loc[df.content_topic == 0]
        df['in_group'] = 0
        df.loc[df['creator_group'] == df['agent_group'],'in_group'] = 1
        df = df.groupby(['time','in_group']).interaction_count.mean().reset_index()
        df['seed'] = seed
        df_all = pd.concat([df_all,df])

    df_all = df_all.groupby(['time','in_group']).interaction_count.mean().reset_index()
    df = df_all
    window_size = 1000
    df = df.sort_values(by=['in_group', 'time'])
    df['interaction_count_mavg'] = df.groupby('in_group')['interaction_count'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean())
    df = df.loc[df.time > 2500]
    plt.plot(df.loc[df.in_group == 1, 'time'].values, df.loc[df.in_group == 1, 'interaction_count_mavg'].values, label=f'{graph_type} (in-group)', color=colors[graph_type], linestyle = 'solid')
    plt.plot(df.loc[df.in_group == 0, 'time'].values, df.loc[df.in_group == 0, 'interaction_count_mavg'].values, label=f'{graph_type} (cross-group)', color=colors[graph_type], linestyle = '--')

plt.legend(title='Network structure')
plt.xlabel('Time')
plt.ylabel('Smoothed interaction count')
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_inter_count_rec_0.png'))
plt.clf()