"""
Finding 2: Plot results over several runs, averaged
"""

import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import seaborn as sns
from scipy.stats import pearsonr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
recommendation_policies = ['real_graph']
init_edges = ['homophilic']
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

# Plot functions
def get_linear(x,y):
    # Get linear regression approximations, returns y_hat
    slope, intercept = np.polyfit(x.values, y.values, 1)
    return slope * x + intercept


for recommendation_policy in recommendation_policies:
    results_path_policy = results_path + f'/{recommendation_policy}/'
    plot_df = pd.DataFrame()

    for graph_type in init_edges:
        df_all = pd.DataFrame()

        for seed in seeds:
            dir_path = os.path.join(results_path_policy, f'policy_{recommendation_policy}_agents_{args.num_agents}_timesteps_{args.time_steps}_init_{graph_type}_groups_{str(group_shares)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seed_{seed}')
            df_agent = pd.read_csv(os.path.join(dir_path,'res_agent.csv'),index_col=False)

            # Convert to averages
            df_agent['recommended_created_content_0'] = (df_agent['recommended_created_content_to0_0'] + df_agent['recommended_created_content_to1_0'])/df_agent['created_content_0']
            df_agent['recommended_created_content_1'] = (df_agent['recommended_created_content_to0_1'] + df_agent['recommended_created_content_to1_1'])/ df_agent['created_content_1']
            df_agent['recommended_created_content_2'] = (df_agent['recommended_created_content_to0_2'] + df_agent['recommended_created_content_to1_2'])/ df_agent['created_content_2']
            df_agent['recommended_created_content_0'].fillna(0, inplace=True)
            df_agent['recommended_created_content_1'].fillna(0, inplace=True)
            df_agent['recommended_created_content_2'].fillna(0, inplace=True)

            df_agent['recommended_created_content_to0_0'] = df_agent['recommended_created_content_to0_0']/df_agent['created_content_0']
            df_agent['recommended_created_content_to0_1'] = df_agent['recommended_created_content_to0_1']/ df_agent['created_content_1']
            df_agent['recommended_created_content_to0_2'] = df_agent['recommended_created_content_to0_2']/ df_agent['created_content_2']
            df_agent['recommended_created_content_to0_0'].fillna(0, inplace=True)
            df_agent['recommended_created_content_to0_1'].fillna(0, inplace=True)
            df_agent['recommended_created_content_to0_2'].fillna(0, inplace=True)

            df_agent['recommended_created_content_to1_0'] = df_agent['recommended_created_content_to1_0']/df_agent['created_content_0']
            df_agent['recommended_created_content_to1_1'] = df_agent['recommended_created_content_to1_1']/ df_agent['created_content_1']
            df_agent['recommended_created_content_to1_2'] = df_agent['recommended_created_content_to1_2']/ df_agent['created_content_2']
            df_agent['recommended_created_content_to1_0'].fillna(0, inplace=True)
            df_agent['recommended_created_content_to1_1'].fillna(0, inplace=True)
            df_agent['recommended_created_content_to1_2'].fillna(0, inplace=True)

            df_agent['incoming_edges'] = df_agent['incoming_edges_from_group_0'] + df_agent['incoming_edges_from_group_1']
            df_agent['outgoing_edges'] = df_agent['outgoing_edges_to_group_0'] + df_agent['outgoing_edges_to_group_1']
            df_all = pd.concat([df_all, df_agent])

        df = df_all

        # Incoming edges vs overall recommendations
        ax = plt.figure(figsize=(5, 4))
        alpha = 0.01
        plt.scatter(df.loc[df.group == 0, 'incoming_edges'], df.loc[df.group == 0, 'recommended_created_content_0'], alpha=alpha)
        plt.plot(df.loc[df.group == 0, 'incoming_edges'], get_linear(df.loc[df.group == 0, 'incoming_edges'], df.loc[df.group == 0, 'recommended_created_content_0']), label='Minority group')
        plt.scatter(df.loc[df.group == 1, 'incoming_edges'], df.loc[df.group == 1, 'recommended_created_content_0'], alpha=alpha)
        plt.plot(df.loc[df.group == 1, 'incoming_edges'], get_linear(df.loc[df.group == 1, 'incoming_edges'], df.loc[df.group == 1, 'recommended_created_content_0']), label='Majority group')
        plt.xlabel('Incoming edges')
        plt.ylabel('Avg. rec. professional topic')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_incoming_edges_rec_1.png'))
        plt.clf()

        ax = plt.figure(figsize=(5, 4))
        plt.scatter(df.loc[df.group == 0, 'incoming_edges'], df.loc[df.group == 0, 'recommended_created_content_1'], alpha=alpha)
        plt.plot(df.loc[df.group == 0, 'incoming_edges'], get_linear(df.loc[df.group == 0, 'incoming_edges'], df.loc[df.group == 0, 'recommended_created_content_1']), label='Minority group')
        plt.scatter(df.loc[df.group == 1, 'incoming_edges'], df.loc[df.group == 1, 'recommended_created_content_1'], alpha=alpha)
        plt.plot(df.loc[df.group == 1, 'incoming_edges'], get_linear(df.loc[df.group == 1, 'incoming_edges'], df.loc[df.group == 1, 'recommended_created_content_1']), label='Majority group')
        plt.xlabel('Incoming edges')
        plt.ylabel('Avg. rec. mainstream  topic')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_incoming_edges_rec_2.png'))
        plt.clf()

        ax = plt.figure(figsize=(5, 4))
        plt.scatter(df.loc[df.group == 0, 'incoming_edges'], df.loc[df.group == 0, 'recommended_created_content_2'], alpha=alpha)
        plt.plot(df.loc[df.group == 0, 'incoming_edges'], get_linear(df.loc[df.group == 0, 'incoming_edges'], df.loc[df.group == 0, 'recommended_created_content_2']), label='Minority group')
        plt.scatter(df.loc[df.group == 1, 'incoming_edges'], df.loc[df.group == 1, 'recommended_created_content_2'], alpha=alpha)
        plt.plot(df.loc[df.group == 1, 'incoming_edges'], get_linear(df.loc[df.group == 1, 'incoming_edges'], df.loc[df.group == 1, 'recommended_created_content_2']), label='Majority group')
        plt.xlabel('Incoming edges')
        plt.ylabel('Avg. rec. marginal topic')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_incoming_edges_rec_3.png'))
        plt.clf()

        # Follower share for minority group
        df_temp = df.loc[df.group == 0]
        temp = df_temp['incoming_edges_from_group_0'] / df_temp['incoming_edges']
        print(1 - temp.mean())
        print(temp.std())

        # Work content recommendation share for minority group
        df_temp = df.loc[df.group == 0]
        temp = df_temp['recommended_created_content_to0_0'] / df_temp['recommended_created_content_0']
        print(1-temp.mean())
        print(temp.std())

        # Minority work content rec to majority group -- edges
        df_temp = df.loc[df.group == 0]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        alpha = 0.05
        axs[0].scatter(df_temp['incoming_edges_from_group_1'], df_temp['recommended_created_content_to1_0'], alpha=alpha)
        axs[0].plot(df_temp['incoming_edges_from_group_1'], get_linear(df_temp['incoming_edges_from_group_1'], df_temp['recommended_created_content_to1_0']))
        axs[0].set_xlabel('Incoming edges from maj. group')
        axs[0].set_ylabel('Avg. rec. work-related content \nto maj. group')
        axs[0].grid()
        axs[1].scatter(df_temp['outgoing_edges_to_group_1'], df_temp['recommended_created_content_to1_0'], alpha=alpha)
        axs[1].plot(df_temp['outgoing_edges_to_group_1'], get_linear(df_temp['outgoing_edges_to_group_1'], df_temp['recommended_created_content_to1_0']))
        axs[1].set_xlabel('Outgoing edges to maj. group')
        axs[1].set_ylabel('Avg. rec. work-related content \nto maj. group')
        axs[1].grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_minority_work_rec_to_maj_edges.png'))
        plt.clf()

        # Minority work content rec to majority group -- preferences
        df_temp = df.loc[df.group == 0]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        alpha = 0.05
        axs[0].scatter(df_temp['pref_topic_1'], df_temp['recommended_created_content_to1_0'], alpha=alpha)
        axs[0].plot(df_temp['pref_topic_1'], get_linear(df_temp['pref_topic_1'], df_temp['recommended_created_content_to1_0']))
        axs[0].set_xlabel('Preference work-related content')
        axs[0].set_ylabel('Avg. rec. work-related content \nto maj. group')
        axs[0].grid()
        axs[1].scatter(df_temp['pref_topic_2'], df_temp['recommended_created_content_to1_0'], alpha=alpha)
        axs[1].plot(df_temp['pref_topic_2'], get_linear(df_temp['pref_topic_2'], df_temp['recommended_created_content_to1_0']))
        axs[1].set_xlabel('Preference majority-topic content')
        axs[1].set_ylabel('Avg. rec. work-related content \nto maj. group')
        axs[1].grid()
        axs[2].scatter(df_temp['pref_topic_3'], df_temp['recommended_created_content_to1_0'], alpha=alpha)
        axs[2].plot(df_temp['pref_topic_3'], get_linear(df_temp['pref_topic_3'], df_temp['recommended_created_content_to1_0']))
        axs[2].set_xlabel('Preference minority-topic content')
        axs[2].set_ylabel('Avg. rec. work-related content \nto maj. group')
        axs[2].grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_minority_work_rec_to_maj_pref.png'))
        plt.clf()

        # Get correlations
        test_result = pearsonr(df_temp['recommended_created_content_to1_0'], df_temp["incoming_edges_from_group_1"])
        print(test_result) #p<0.001
        test_result = pearsonr(df_temp['recommended_created_content_to1_0'], df_temp["outgoing_edges_to_group_1"])
        print(test_result) #p<0.05
        test_result = pearsonr(df_temp['recommended_created_content_to1_0'], df_temp["pref_topic_1"])
        print(test_result) #p<0.001
        print(test_result) = pearsonr(df_temp['recommended_created_content_to1_0'], df_temp["pref_topic_2"])
        print(test_result) #p<0.001
        test_result = pearsonr(df_temp['recommended_created_content_to1_0'], df_temp["pref_topic_3"])
        print(test_result) #p<0.001
        print(df_temp.corr()['recommended_created_content_to1_0'])

        # Edges within and across groups
        df_temp = df.groupby('group')[['outgoing_edges_to_group_0', 'outgoing_edges_to_group_1']].sum().reset_index()
        print(df_temp.loc[(df_temp.group == 1),'outgoing_edges_to_group_0'].values / (df_temp.loc[(df_temp.group == 0),'outgoing_edges_to_group_0'].values + df_temp.loc[(df_temp.group == 1),'outgoing_edges_to_group_0'].values))
        within_group = df_temp.loc[df_temp.group == 0,'outgoing_edges_to_group_0'].values + df_temp.loc[df_temp.group == 1,'outgoing_edges_to_group_1'].values
        across_group = df_temp.loc[df_temp.group == 1, 'outgoing_edges_to_group_0'].values + df_temp.loc[df_temp.group == 0, 'outgoing_edges_to_group_1'].values
        print(within_group[0] / (within_group[0] + across_group[0]))
        print(across_group[0] / (within_group[0] + across_group[0]))

        plt.figure(figsize=(5, 4))
        sns.kdeplot(x='outgoing_edges_to_group_0', data=df.loc[df.group == 0], label='To min. grou')
        sns.kdeplot(x='outgoing_edges_to_group_1', data=df.loc[df.group == 0], label='To maj. group')
        plt.xlabel('Outgoing edges')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_kde0.png'))
        plt.clf()

        plt.figure(figsize=(5, 4))
        sns.kdeplot(x='outgoing_edges_to_group_0', data=df.loc[df.group == 1], label='To min. grou')
        sns.kdeplot(x='outgoing_edges_to_group_1', data=df.loc[df.group == 1], label='To maj. group')
        plt.xlabel('Outgoing edges')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_kde1.png'))
        plt.clf()


## Density plot of topic preferences
recommendation_policy = 'real_graph'
graph_type = 'homophilic'
df_all = pd.DataFrame()
results_path_policy = results_path + f'/{recommendation_policy}/'

for seed in seeds:
    dir_path = os.path.join(results_path_policy, f'policy_{recommendation_policy}_agents_{args.num_agents}_timesteps_{args.time_steps}_init_{graph_type}_groups_{str(group_shares)}_b_{b_str}_alpha_{args.alpha}_p_{p_str}_seed_{seed}')
    df_agent = pd.read_csv(os.path.join(dir_path,'res_agent.csv'),index_col=False)
    df_all = pd.concat([df_all,df_agent])

df = df_all
d = {0: 'work-related', 1:'mainstream-topic', 2:'marginal-topic'}

for i, column in enumerate(['pref_topic_1', 'pref_topic_2', 'pref_topic_3']):
    ax = plt.figure(figsize=(5, 4))
    if i < 2:
        sns.histplot(data=df, x=column, hue='group', legend=False)
    else:
        sns.histplot(data=df, x=column, hue='group')
    plt.grid()
    plt.xlabel(f'Preference {d[i]} content')
    plt.ylabel('Count')
    if i == 2:
        plt.legend(title='', labels=['Majority group', 'Minority group'])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_pref_hist_{i}.png'))

# Plot network sizes
ax = plt.figure(figsize=(5, 4))
sns.histplot(data=df, x='outgoing_edges_to_group_1', hue='group', legend=False)
#plt.legend(title='', labels=['Majority group', 'Minority group'])
plt.grid()
plt.xlabel(f'Edges to majority group')
plt.ylabel('Count')
plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_edges_maj.png'))

ax = plt.figure(figsize=(5, 4))
sns.histplot(data=df, x='outgoing_edges_to_group_0', hue='group', legend=True)
plt.legend(title='', labels=['Majority group', 'Minority group'])
plt.grid()
plt.xlabel(f'Edges to minority group')
plt.ylabel('Count')
plt.savefig(os.path.join(plot_path, f'{recommendation_policy}_{graph_type}_edges_min.png'))




