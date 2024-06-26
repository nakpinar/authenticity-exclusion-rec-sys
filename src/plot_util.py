"""
Plot function for single simulation runs
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_number_content_time(df_timestep):
    """
    Number of content items creates and recommendations made over time
    """

    plt.plot(df_timestep.timestep, df_timestep.number_content, label='Created content')
    plt.plot(df_timestep.timestep, df_timestep.number_recommendations, label='Recommendations')
    plt.xlabel("Time step")
    plt.ylabel("Number items")
    plt.title("Number of items created and number of recommendations made over time")
    plt.legend()
    
    return plt

def plot_interaction_count_recommendation(df_recommendation, window_size=100):
    """
    Average interaction count among the recommendation pairs
    """

    df = df_recommendation.groupby('time').interaction_count.mean().rolling(window_size,1).mean().reset_index()

    plt.plot(df['time'], df['interaction_count'])
    plt.xlabel("Time step")
    plt.ylabel("Average smoothed interaction count")
    plt.title(f"Rolling (ws = {window_size}) average interaction count between user pairs that are part of a recommendation")

    return plt

def plot_topic_dist_content(df_topic):
    """
    Topic distribution among created and recommended content
    """

    df_topic['content_count'] = df_topic['content_count'] / df_topic['content_count'].sum()
    df_topic['recommendation_count'] = df_topic['recommendation_count'] / df_topic['recommendation_count'].sum()

    width = 0.35
    fig,ax = plt.subplots()
    rects1 = ax.bar(df_topic.topic - width/2, df_topic.content_count, width, label='Created content')
    rects2 = ax.bar(df_topic.topic + width/2, df_topic.recommendation_count, width, label='Recommendations')
    ax.set_title('Topic distribution among created and recommended content')
    ax.set_xlabel('Topics')
    ax.set_ylabel('Share')
    ax.set_xticks(np.arange(len(df_topic.topic)))
    ax.set_xticklabels(df_topic.topic)
    ax.legend()
    
    return plt

def plot_interaction_count_time(df_timestep):
    """
    Interaction count over time
    """

    plt.plot(df_timestep.timestep, df_timestep.in_group_interaction_count, label='Within groups')
    plt.plot(df_timestep.timestep, df_timestep.out_group_interaction_count, label='Across groups')
    plt.plot(df_timestep.timestep, df_timestep.interaction_count, label='Overall')
    plt.xlabel("Time step")
    plt.ylabel("Count")
    plt.title("Average smoothed interaction count")
    plt.legend()

    return plt

def plot_number_recommendation_topic(df_content, topic=0, window_size=100, type='mean'):
    """
    Number of times a piece of content from topic is recommended over time by creator group, MAVG with window_size
    """

    df_content = df_content.loc[df_content.topic == topic]

    if type == 'mean':
        rolling0 = df_content.loc[df_content.creator_group == 0].groupby('time')['count'].mean().rolling(window_size,1).mean().reset_index()
        rolling1 = df_content.loc[df_content.creator_group == 1].groupby('time')['count'].mean().rolling(window_size,1).mean().reset_index()
    elif type == 'median':
        rolling0 = df_content.loc[df_content.creator_group == 0].groupby('time')['count'].median().rolling(window_size,1).mean().reset_index()
        rolling1 = df_content.loc[df_content.creator_group == 1].groupby('time')['count'].median().rolling(window_size,1).mean().reset_index()
    else:
        raise NotImplementedError

    plt.plot(rolling0['time'], rolling0['count'], label='Group 0')
    plt.plot(rolling1['time'], rolling1['count'], label='Group 1')
    plt.xlabel("Time step")
    plt.ylabel("Average recommendations")
    plt.title(f"Rolling (ws = {window_size}) number of times content of topic {topic} is recommeded by creator group")
    plt.legend()
    
    return plt

def plot_diff_number_recommendation_topic(df_content, topic=0, window_size=100, type='mean'):
    """
    Difference between the number of times a piece of content from topic is recommended over time by creator group, MAVG with window_size
    """

    df_content = df_content.loc[df_content.topic == topic]

    if type == 'mean':
        mean0 = df_content.loc[df_content.creator_group == 0].groupby('time')['count'].mean().reset_index(name='mean0')
        mean1 = df_content.loc[df_content.creator_group == 1].groupby('time')['count'].mean().reset_index(name='mean1')
        mean = mean0.merge(mean1, on='time', how='outer')
        mean = mean.fillna(0)
        mean['diff'] = mean['mean1'] - mean['mean0']
        mean = mean.sort_values(by='time')
        mean['diff_roll'] = mean['diff'].rolling(window_size,1).mean()
    else:
        raise NotImplementedError


    plt.plot(mean['time'], mean['diff_roll'])
    plt.xlabel("Time step")
    plt.ylabel(f"Difference in average number of recommendation, rolling {window_size}")
    plt.title(f"Difference (group 1 - 0) number of times content of topic {topic} is recommeded")
    plt.legend()

    return plt
