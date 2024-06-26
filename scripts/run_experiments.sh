#!/bin/bash

# Define the group0_share values and seeds in arrays
init_edges=('complete' 'random' 'homophilic')
recommendation_policies=('real_graph' 'random' 'topic')
seeds=({0..9})


# Loop through each combination of group0_share and seed
for init_edge in "${init_edges[@]}"; do
    for recommendation_policy in "${recommendation_policies[@]}"; do
        for seed in "${seeds[@]}"; do
            python run_simulation.py --recommendation_policy="$recommendation_policy" --group0_share=0.2 --init_edges="$init_edge" --time_steps=10000 --b0=1. --b1=1. --b2=5. --b3=-1. --p0=0.5 --p1=0.4 --p01=0.1 --p10=0.1 --alpha=0.01 --seed="$seed" &
        done
    done
done

# Wait for all background processes to finish
wait