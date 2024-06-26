
"""
Agent Based Simulation components 
"""

import numpy as np
import pandas as pd
import time
import os

def standardize_matrix(matrix, zero_diagonal=True):
    """
    Standardize matrix to have mean 0 and std 1.
    If zero_diagonal, diagonal is ignored and set to 0.
    matrix: numpy.array
    """
    std_matrix = matrix.copy()
    if zero_diagonal:
        mask = ~np.eye(std_matrix.shape[0], dtype=bool) # Selects everything except diagonal
        m = np.mean(std_matrix[mask])
        s = np.std(std_matrix[mask])

        if np.abs(s) < 1e-5: # Constant case
            std_matrix[mask] = std_matrix[mask] - m    
        else:
            std_matrix[mask] = (std_matrix[mask] - m) /s
        
        np.fill_diagonal(std_matrix,0)
    else:
        if np.abs(std_matrix.std()) < 1e-5:
            std_matrix = std_matrix - std_matrix.mean()
        else:
            std_matrix = (std_matrix - std_matrix.mean()) / std_matrix.std()
    
    return std_matrix


class Agent:
    """
    Base class for all agents in the simulation.
    """
    def __init__(self, unique_id, model):
        """
        Initialize a new agent with a unique ID, a reference to the simulation model, and a set of content preferences.
        """
        self.unique_id = unique_id
        self.model = model

        # Sample group membership
        self.group = np.random.choice(len(self.model.group_shares), p=self.model.group_shares)

        # Sample vector of topic preferences based on group, sums to 1 
        self.preferences = self.model.preference_means[self.group] + np.random.multivariate_normal(mean=np.zeros(3), cov=self.model.preference_cov)
        self.preferences[self.preferences < 0] = 0
        self.preferences = self.preferences / self.preferences.sum()

        # Set content creation and recommendation seeking probabilities
        self.content_prob = 0.5
        self.seek_recommendation_prob = 0.5

        # Number of content created and total number that content has been recommended by topic
        self.created_content = [0, 0, 0]
        self.recommended_created_content = {'0': [0, 0, 0],'1': [0, 0, 0]} # By group of the agent it is recommended to
    
    def create_content(self, time):
        """
        Create new content with probability and add it to the model.
        """
        if np.random.rand() < self.content_prob:
            self.model.content_counter += 1
            content_counter = self.model.content_counter
            content = Content(self,time,content_counter)
            self.model.content[time].append(content)
            self.created_content[content.topic] += 1

    def seek_recommendation(self):
        """
        Return whether user seeks recommendation in this time step
        """
        return np.random.rand() < self.seek_recommendation_prob


class Content:
    """
    Base class for all content in the simulation.
    """
    def __init__(self, creator, time, unique_id):
        """
        Initialize a new content item with a reference to its creator, a topic, and a time stamp for the time of creation. 
        """
        self.unique_id = unique_id
        self.creator = creator
        self.time = time
        self.times_recommended = 0

        # Sample topic from preferences
        self.topic = np.random.choice(len(self.creator.preferences), p=self.creator.preferences)
    

class Model:
    """
    The main model class, which initializes the simulation and runs the agent-based model.
    """
    def __init__(self,
                 recommendation_policy='real_graph',
                 num_agents=1000,
                 group_shares=np.array([0.7,0.3]),
                 preference_means=np.array([[5,1,4],[5,4,1]]),
                 preference_cov=np.eye(3)/10,
                 init_edges='complete',
                 alpha=0.05,
                 b=np.array([1,1,1,-1]),
                 p=np.array([-1,-1,-1,-1]),
                 dir_path=False,
                 write_csv_timesteps=1000,
                 time_simulation=False):
        """
        Initialize the simulation with a specified number of agents and an empty set of content.
        """
        self.recommendation_policy = recommendation_policy
        self.num_agents = num_agents
        self.group_shares = group_shares
        self.preference_means = preference_means
        self.preference_cov = preference_cov
        self.init_edges = init_edges
        self.agents = []
        self.content = dict() 
        self.recommendations = dict() 
        self.edge_weights = dict()
        self.smoothed_interaction_count = dict()
        self.time = 0 
        self.content_counter = 0
        self.alpha = alpha # Discounting factor for interaction count MAVG smoothing
        self.b = b # Parameters for edge weight model
        self.p = p # Parameters for stochastic block model (only for init_edges = homophilic)
        self.time_simulation = time_simulation
        self.runtimes = dict()
        self.dir_path = dir_path
        self.write_csv_timesteps = write_csv_timesteps
            
        if self.time_simulation:
            t0 = time.time()

        # Create results folder
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # Create agents
        for i in range(self.num_agents):
            agent = Agent(i, self)
            self.agents.append(agent)
        
        # Add edges between agents, edges are directed
        if self.init_edges == 'complete':
            self.adj_matrix = np.ones((self.num_agents, self.num_agents))
        elif self.init_edges == 'random':
            p = 0.5
            self.adj_matrix = (np.random.rand(self.num_agents, self.num_agents) < p) * 1
        elif self.init_edges == 'homophilic': #Stochastic block model
            # Create a meshgrid of indices
            i, j = np.meshgrid(np.arange(self.num_agents), np.arange(self.num_agents), indexing='ij')
            p_ij = np.zeros((self.num_agents, self.num_agents))

            # Set values based on conditions
            agent_groups = np.array([agent.group for agent in self.agents])
            p_ij[(agent_groups[i] == 0) & (agent_groups[j] == 0)] = self.p[0]
            p_ij[(agent_groups[i] == 1) & (agent_groups[j] == 1)] = self.p[1]
            p_ij[(agent_groups[i] == 0) & (agent_groups[j] == 1)] = self.p[2]
            p_ij[(agent_groups[i] == 1) & (agent_groups[j] == 0)] = self.p[3]

            self.adj_matrix = (np.random.rand(self.num_agents, self.num_agents) < p_ij) * 1
        else:
            raise NotImplementedError

        # Number of accounts both i and j follow
        self.common_outgoing_edges = np.dot(self.adj_matrix, self.adj_matrix.T)
        np.fill_diagonal(self.common_outgoing_edges,0)

        # Number of accounts that follow both i and j
        self.common_incoming_edges = np.dot(self.adj_matrix.T, self.adj_matrix)
        np.fill_diagonal(self.common_incoming_edges,0)

        # Compute topic preference distance for agents, Euclidean distance
        self.dist_matrix = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            for j in range(i+1,self.num_agents):
                self.dist_matrix[i][j] = np.sqrt(np.sum((self.agents[i].preferences - self.agents[j].preferences)**2))
                self.dist_matrix[j][i] = self.dist_matrix[i][j]

        # Initialize interaction features and edge weights
        self.smoothed_interaction_count[0] = np.zeros((self.num_agents, self.num_agents))
        self.edge_weights[0] = self.get_edge_weights()

        if self.time_simulation:
            self.runtimes['Initialize model'] = time.time() - t0
                
    def content_step(self):
        """
        Simulate content creation step according to each agent's content creation probability and topic preferences.
        """
        self.content[self.time] = []
        for agent in self.agents:
            agent.create_content(self.time)

    def follows(self, agent1, agent2):
        """
        Return True if agent1 follows agent2, else False.
        """
        if np.abs(self.adj_matrix[agent1.unique_id][agent2.unique_id]-1) < 1e-5:
            return True
        return False

    def recommendation_step(self):
        """
        Go over all the agents and (1) sample whether they seek out recommendation in this time step, and (2) what the recommendation is 
        """
        if self.recommendation_policy == 'real_graph':

            # Gather all the recommendations for this time step
            self.recommendations[self.time] = []
            for agent in self.agents: 
                if agent.seek_recommendation():

                    # Gather candidates tuples of (content,creator)
                    candidate_content = [item for item in self.content[self.time] if self.follows(agent,item.creator)]

                    # Quit loop in case there is no content to recommend
                    if len(candidate_content) == 0:
                        break

                    # Get scores for candidates
                    candidate_edge_weights = np.array([self.edge_weights[self.time - 1][agent.unique_id,item.creator.unique_id] for item in candidate_content])
                    candidate_topic_score = np.array([agent.preferences[item.topic] for item in candidate_content])

                    scores = (candidate_edge_weights + candidate_topic_score) / 2
                    if len(scores) > 1:
                        scores = scores - scores.min()

                    # Sample recommended item according to scores
                    probs = scores / scores.sum()
                    idx = np.random.choice(np.arange(len(probs)), p=probs)
                    recommended_item = candidate_content[idx]
                    recommended_item.times_recommended += 1

                    # Keep track of smoothed interaction counts among the recommendation pairs
                    inter_count = self.smoothed_interaction_count[self.time - 1][agent.unique_id,recommended_item.creator.unique_id]

                    self.recommendations[self.time].append([agent,recommended_item,inter_count])
                    recommended_item.creator.recommended_created_content[str(agent.group)][recommended_item.topic] += 1
        
        elif self.recommendation_policy == 'edge_weight':

            # Gather all the recommendations for this time step
            self.recommendations[self.time] = []
            for agent in self.agents: 
                if agent.seek_recommendation():

                    # Gather candidates tuples of (content,creator)
                    candidate_content = [item for item in self.content[self.time] if self.follows(agent,item.creator)]

                    # Quit loop in case there is no content to recommend
                    if len(candidate_content) == 0:
                        break

                    # Get scores for candidates -- only edge weights
                    candidate_edge_weights = np.array([self.edge_weights[self.time - 1][agent.unique_id,item.creator.unique_id] for item in candidate_content])
                    scores = candidate_edge_weights
                    if len(scores) > 1:
                        scores = scores - scores.min()

                    # Sample recommended item according to scores
                    probs = scores / scores.sum()
                    idx = np.random.choice(np.arange(len(probs)), p=probs)
                    recommended_item = candidate_content[idx]
                    recommended_item.times_recommended += 1

                    # Keep track of smoothed interaction counts among the recommendation pairs
                    inter_count = self.smoothed_interaction_count[self.time - 1][agent.unique_id, recommended_item.creator.unique_id]

                    self.recommendations[self.time].append([agent,recommended_item,inter_count])
                    recommended_item.creator.recommended_created_content[str(agent.group)][recommended_item.topic] += 1

        elif self.recommendation_policy == 'random':

            # Gather all the recommendations for this time step
            self.recommendations[self.time] = []
            for agent in self.agents: 
                if agent.seek_recommendation():

                    # Gather candidates tuples of (content,creator)
                    candidate_content = [item for item in self.content[self.time] if self.follows(agent,item.creator)]

                    # Quit loop in case there is no content to recommend
                    if len(candidate_content) == 0:
                        break

                    # Sample recommended item uniformly
                    probs = [1 / len(candidate_content)] * len(candidate_content) 
                    idx = np.random.choice(np.arange(len(probs)), p=probs)
                    recommended_item = candidate_content[idx]
                    recommended_item.times_recommended += 1

                    self.recommendations[self.time].append([agent,recommended_item,0])
                    recommended_item.creator.recommended_created_content[str(agent.group)][recommended_item.topic] += 1

        elif self.recommendation_policy == 'topic': # Recommendation just dependent on how good the preference topic match is between consumer and content 
            
            # Gather all the recommendations for this time step
            self.recommendations[self.time] = []
            for agent in self.agents: 
                if agent.seek_recommendation():

                    # Gather candidates tuples of (content,creator)
                    candidate_content = [item for item in self.content[self.time] if self.follows(agent,item.creator)]

                    # Quit loop in case there is no content to recommend
                    if len(candidate_content) == 0:
                        break

                    candidate_topic_score = np.array([agent.preferences[item.topic] for item in candidate_content])

                    # Sample according to scores
                    probs = candidate_topic_score / candidate_topic_score.sum()
                    idx = np.random.choice(np.arange(len(probs)), p=probs)
                    recommended_item = candidate_content[idx]
                    recommended_item.times_recommended += 1 

                    self.recommendations[self.time].append([agent,recommended_item,0])
                    recommended_item.creator.recommended_created_content[str(agent.group)][recommended_item.topic] += 1
        
        elif self.recommendation_policy == 'interaction':
            
            # Gather all the recommendations for this time step
            self.recommendations[self.time] = []
            for agent in self.agents: 
                if agent.seek_recommendation():

                    # Gather candidates tuples of (content,creator)
                    candidate_content = [item for item in self.content[self.time] if self.follows(agent,item.creator)]

                    # Quit loop in case there is no content to recommend
                    if len(candidate_content) == 0:
                        break

                    # Get scores for candidates -- only MAVG interaction counts
                    candidate_weights = np.array([self.smoothed_interaction_count[self.time - 1][agent.unique_id,item.creator.unique_id] for item in candidate_content])
                    if candidate_weights.sum() < 1e-5: # No interaction counts in the first step
                        scores = np.ones(len(candidate_weights))
                    else:
                        scores = candidate_weights                    

                    # Sample recommended item according to scores
                    probs = scores / scores.sum()
                    idx = np.random.choice(np.arange(len(probs)), p=probs)
                    recommended_item = candidate_content[idx]
                    recommended_item.times_recommended += 1

                    # Keep track of smoothed interaction counts among the recommendation pairs
                    inter_count = self.smoothed_interaction_count[self.time - 1][agent.unique_id, recommended_item.creator.unique_id]

                    self.recommendations[self.time].append([agent,recommended_item,inter_count])
                    recommended_item.creator.recommended_created_content[str(agent.group)][recommended_item.topic] += 1

        else:
            raise NotImplementedError('Unknown recommendation policy.')
        
    def sample_interaction(self):
        """
        Sample if users interact with recommendations. Interaction probability is exactly the topic match between consumer and recommended item.
        """

        user_pairs = np.array(
            [[agent.unique_id, recommended_item.creator.unique_id, agent.preferences[recommended_item.topic]] for
             agent, recommended_item, _ in self.recommendations[self.time]])

        if len(user_pairs) > 0:
            recommendation_matrix = np.zeros((self.num_agents,self.num_agents))
            recommendation_matrix[tuple(user_pairs[:,[0,1]].astype(int).T)] = 1

            # Interaction probability is exactly the topic match between consumer and recommended item
            interaction_matrix = np.zeros((self.num_agents,self.num_agents))
            flat_indices = np.ravel_multi_index((user_pairs[:,0].astype(int),user_pairs[:,1].astype(int)), interaction_matrix.shape)
            np.put(interaction_matrix,flat_indices,user_pairs[:,2])
            interaction_matrix = (np.random.random(interaction_matrix.shape) < interaction_matrix)

            # # Constant interaction probability
            # interaction_matrix = recommendation_matrix.copy()
            # interaction_prob=0.8
            # interaction_matrix = interaction_matrix * (np.random.rand(self.num_agents,self.num_agents) < interaction_prob)

        else:
            recommendation_matrix = np.zeros((self.num_agents, self.num_agents))
            interaction_matrix = np.zeros((self.num_agents, self.num_agents))

        return recommendation_matrix, interaction_matrix

    def update_step(self, recommendation_matrix, interaction_matrix):
        """
        Update model parameters
        """
        # Update smoothed interaction count. This is conditional on recommendation, i.e. only the counts for which a recommendation has been made in this step are updated. 
        updated_smoothed_interaction_count = self.smoothed_interaction_count[self.time - 1].copy()
        updated_smoothed_interaction_count[recommendation_matrix == 1] = interaction_matrix[recommendation_matrix == 1] * self.alpha + updated_smoothed_interaction_count[recommendation_matrix == 1] * (1-self.alpha)
        self.smoothed_interaction_count[self.time] = updated_smoothed_interaction_count

        # Update edge weights, this already uses the new interaction counts 
        self.edge_weights[self.time] = self.get_edge_weights()

    def get_edge_weights(self):
        """
        Get the edge weights of the graph; probability range.
        """
        logodds = self.b[0] * standardize_matrix(self.common_outgoing_edges) \
                + self.b[1] * standardize_matrix(self.common_incoming_edges) \
                + self.b[2] * standardize_matrix(self.smoothed_interaction_count[self.time]) \
                + self.b[3] * standardize_matrix(self.dist_matrix)
        edge_weights = 1/(1+np.exp(-logodds))
        np.fill_diagonal(edge_weights,0)

        return edge_weights

    def run(self, time_steps=1000):
        """
        Run the whole simulation for a number of timesteps.
        """

        print(f'Run simulation with results path {self.dir_path}.')

        if self.time_simulation:
            t_run = time.time()

        if self.time_simulation:
            times_content = []
            times_recommendation = []
            times_update = []

        for t in range(self.time,self.time + time_steps):
            
            self.time += 1
            t0 = time.time()
            self.content_step()
            t1 = time.time()
            self.recommendation_step()
            t2 = time.time()
            recommendation_matrix,interaction_matrix = self.sample_interaction()
            self.update_step(recommendation_matrix,interaction_matrix)
            t3 = time.time()
            
            if self.time_simulation:
                times_content.append(t1 - t0)
                times_recommendation.append(t2 - t1)
                times_update.append(t3 - t2)

            # Write data to file after each self.write_csv_timesteps iterations and free up memory
            if self.time > 0 and (self.time + 1) % self.write_csv_timesteps == 0:

                df = self.get_res_content(all_timesteps=False)
                df.to_csv(os.path.join(self.dir_path, f'res_content_{self.time+1}.csv'), index=False)
                df = self.get_res_recommendation(all_timesteps=False)
                df.to_csv(os.path.join(self.dir_path, f'res_recommendation_{self.time + 1}.csv'), index=False)
                df = self.get_res_timestep(all_timesteps=False)
                df.to_csv(os.path.join(self.dir_path, f'res_timestep_{self.time+1}.csv'), index=False)
                df = self.get_res_topic(all_timesteps=False)
                df.to_csv(os.path.join(self.dir_path, f'res_topic_{self.time+1}.csv'), index=False)

                self.content = dict()
                self.recommendations = dict()
                self.edge_weights = dict({self.time: self.edge_weights[self.time]})
                self.smoothed_interaction_count = dict({self.time: self.smoothed_interaction_count[self.time]})

                print(f'Write data to file after t={self.time}...')

        if self.time_simulation:
            self.runtimes['Average content_step()'] = sum(times_content)/len(times_content)
            self.runtimes['Average recommendation_step()'] = sum(times_recommendation)/len(times_recommendation)
            self.runtimes['Average update_step()'] = sum(times_update)/len(times_update)
            self.runtimes['Average run() call'] = (time.time() - t_run) / time_steps
            self.runtimes['Total run() call'] = time.time() - t_run     

    def get_res_agent(self):
        """
        Get results by agent; read previously saved data
        """

        # Group and topic preferences
        df = pd.DataFrame([[item.unique_id, item.group] + item.preferences.tolist() for item in self.agents],
                          columns=['agent_id', 'group', 'pref_topic_1', 'pref_topic_2', 'pref_topic_3'])

        # Get number of incoming and outgoing edges by group
        groups = np.array([item.group for item in self.agents])

        group_0_mask = groups == 0
        group_1_mask = groups == 1

        edges_to_group_0 = self.adj_matrix[:, group_0_mask]
        edges_to_group_1 = self.adj_matrix[:, group_1_mask]

        df['outgoing_edges_to_group_0'] = np.sum(edges_to_group_0, axis=1)
        df['outgoing_edges_to_group_1'] = np.sum(edges_to_group_1, axis=1)

        adj_matrix_T = self.adj_matrix.T
        incoming_from_group_0 = adj_matrix_T[:, group_0_mask]
        incoming_from_group_1 = adj_matrix_T[:, group_1_mask]

        df['incoming_edges_from_group_0'] = np.sum(incoming_from_group_0, axis=1)
        df['incoming_edges_from_group_1'] = np.sum(incoming_from_group_1, axis=1)

        # Add number of created posts by topic + times their content was recommended by topic
        cc = np.array([item.created_content for item in self.agents])
        df['created_content_0'] = cc[:, 0]
        df['created_content_1'] = cc[:, 1]
        df['created_content_2'] = cc[:, 2]
        cr0 = np.array([item.recommended_created_content['0'] for item in self.agents])
        cr1 = np.array([item.recommended_created_content['1'] for item in self.agents])
        df['recommended_created_content_to0_0'] = cr0[:, 0]
        df['recommended_created_content_to0_1'] = cr0[:, 1]
        df['recommended_created_content_to0_2'] = cr0[:, 2]

        df['recommended_created_content_to1_0'] = cr1[:, 0]
        df['recommended_created_content_to1_1'] = cr1[:, 1]
        df['recommended_created_content_to1_2'] = cr1[:, 2]

        return df
    
    def get_res_content(self, all_timesteps=True):
        """
        Get results by content piece
        """

        df_all = pd.DataFrame()
        df = pd.DataFrame()

        if all_timesteps:
            for t in range(self.write_csv_timesteps, self.time+1, self.write_csv_timesteps):
                df = pd.read_csv(os.path.join(self.dir_path, f'res_content_{t}.csv'))
                df_all = pd.concat([df_all,df], ignore_index=True)


        # Timestep, topic, creator group, number of times content has been recommended
        df = pd.DataFrame(np.array(
            [[val.unique_id, val.time, val.topic, val.creator.group, val.times_recommended] for sublist in
             self.content.values() for val in sublist]), columns=['content_id', 'time', 'topic', 'creator_group', 'count'])

        df_all = pd.concat([df_all,df], ignore_index=True)
        
        return df_all

    def get_res_recommendation(self, all_timesteps=True):
        """
        Get results by recommendation
        """

        df_all = pd.DataFrame()
        df = pd.DataFrame()

        if all_timesteps:
            for t in range(self.write_csv_timesteps, self.time + 1, self.write_csv_timesteps):
                df = pd.read_csv(os.path.join(self.dir_path, f'res_recommendation_{t}.csv'))
                df_all = pd.concat([df_all, df], ignore_index=True)

        # Timestep, agent group, creator group, content topic, interaction count for all the recommendations that have been made
        df = pd.DataFrame(np.array(
            [[val[1].time, val[0].group, val[1].creator.group, val[1].topic, val[2]] for sublist in self.recommendations.values() for val in sublist]),
            columns=['time','agent_group','creator_group','content_topic','interaction_count'])

        df_all = pd.concat([df_all, df], ignore_index=True)

        return df_all

    def get_res_timestep(self, all_timesteps=True):
        """
        Get results by timestep
        """

        df_all = pd.DataFrame()
        df = pd.DataFrame()

        if all_timesteps:
            for t in range(self.write_csv_timesteps, self.time+1, self.write_csv_timesteps):
                df = pd.read_csv(os.path.join(self.dir_path, f'res_timestep_{t}.csv'))
                df_all = pd.concat([df_all,df], ignore_index=True)

        # Number of items created
        times = list(self.content.keys())
        number_content = [len(self.content[k]) for k in self.content.keys()]
        if min(times) < self.write_csv_timesteps - 1:
            times = [0] + times
            number_content = [0] + number_content
        df = pd.DataFrame(np.array([times,number_content]).T,columns=['timestep','number_content'])

        # Number of recommendations
        times = list(self.recommendations.keys())
        number_recommendations = [len(self.recommendations[k]) for k in self.recommendations.keys()]
        if min(times) < self.write_csv_timesteps - 1:
            times = [0] + times
            number_recommendations = [0] + number_recommendations
        df_temp = pd.DataFrame(np.array([times,number_recommendations]).T,columns=['timestep','number_recommendations'])
        df = pd.merge(df,df_temp,on='timestep',how='outer')
        df.fillna({'number_content': 0, 'number_recommendations': 0}, inplace=True)

        # Interaction count within and in between groups
        groups = np.array([item.group for item in self.agents])
        same_groups = (groups[:, None] == groups)
        np.fill_diagonal(same_groups,False)
        opposite_groups = ~same_groups
        np.fill_diagonal(opposite_groups,False)

        in_group_interaction_count = np.array([self.smoothed_interaction_count[t][same_groups].mean() for t in range(self.time - self.time % self.write_csv_timesteps, self.time + 1)])
        out_group_interaction_count = np.array([self.smoothed_interaction_count[t][opposite_groups].mean() for t in range(self.time - self.time % self.write_csv_timesteps, self.time + 1)])
        interaction_count = np.array([self.smoothed_interaction_count[t][~np.eye(self.smoothed_interaction_count[t].shape[0], dtype=bool)].mean() for t in range(self.time - self.time % self.write_csv_timesteps, self.time + 1)])

        df['interaction_count'] = interaction_count
        df['in_group_interaction_count'] = in_group_interaction_count
        df['out_group_interaction_count'] = out_group_interaction_count

        df_all = pd.concat([df_all,df], ignore_index=True)

        return df_all

    def get_res_topic(self, all_timesteps=True):
        """
        Get results by topics (aggregated over timesteps)
        """

        df_all = pd.DataFrame()
        df = pd.DataFrame()

        if all_timesteps:
            for t in range(self.write_csv_timesteps, self.time+1, self.write_csv_timesteps):
                df = pd.read_csv(os.path.join(self.dir_path, f'res_topic_{t}.csv'))
                df_all = pd.concat([df_all,df], ignore_index=True)


        # Number of content items created by topic
        content_topics = np.array([val.topic for sublist in self.content.values() for val in sublist])
        content_topic_counts = np.unique(content_topics, return_counts=True)
        df = pd.DataFrame(content_topic_counts).T
        df.columns = ['topic', 'content_count']

        # Number of recommendations made by topic
        recommendation_topics = np.array(
            [val[1].topic for sublist in self.recommendations.values() for val in sublist])
        recommendation_topic_counts = np.unique(recommendation_topics, return_counts=True)
        df_temp = pd.DataFrame(recommendation_topic_counts).T
        df_temp.columns = ['topic', 'recommendation_count']
        df = pd.merge(df, df_temp, on='topic', how='outer')
        df.fillna({'content_count': 0, 'recommendation_count': 0}, inplace=True)

        df_all = pd.concat([df_all,df], ignore_index=True)

        df_all = df_all.groupby('topic').sum().reset_index()

        return df_all

