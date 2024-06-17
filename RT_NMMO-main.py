import importlib
import nmmo
import torch
from torch import nn
from nmmo.render.replay_helper import FileReplayHelper
import numpy as np
from math import ceil, sqrt
import random

importlib.import_module('RT_NMMO-tools')
importlib.import_module('RT_NMMO-Neural')

# Getting spawn positions
spawn_positions = []
for i in range(player_N):
  spawn_positions.append(env.realm.players.entities[i+1].spawn_pos)

# Setting up a visited tiles dictionary
visited_tiles = {i+1: [] for i in range(player_N)}

# Setting up the average lifetime dictionary
avg_lifetime = {}

# Set up a list of all visited tiles by all agents
all_visited = []

# Set up max lifetime
max_lifetime = 0
max_lifetime_dict = {}

life_durations = {i+1: 0 for i in range(env.config.PLAYER_N)}

steps = 100

for i in range(steps):
  # Uncomment for saving replays
  print(i)
  #if i%1000 == 0:
  #  replay_file = f"/content/replay1"
  #  replay_helper.save(replay_file, compress=True)

  current_oldest = life_durations[max(life_durations, key=life_durations.get)]
  if current_oldest > max_lifetime:
    max_lifetime = current_oldest

  # Assign the top-all-time age record to the current tick
  max_lifetime_dict[i] = max_lifetime

  if i%10000 == 0:
    obs = env.reset()

  elif i%1000 == 0:
    save_state(model_dict, f"weights_128_dynamic")
    !tar chvfz weights_128_dynamic.tar.gz weights_128_dynamic/*

  #If the number of agents alive doesn't correspond to PLAYER_N, respawn
  if env.num_agents != player_N:
    for i in range(player_N):
      if i+1 not in env.realm.players.entities:

        # Spawn individual at a random location
        env.realm.players.cull()
        x, y = random.choice(spawn_positions)
        env.realm.players.spawn_individual(x, y, i+1)

        '''
        # Pick the parents
        parent = pick_best()
        parent2 = 0
        while(parent2 == parent or parent2 == 0):
          parent2 = pick_best()
        '''

        parent = pick_best()

        # Upon the "birth" of a new agent, reset the life duration and visited tiles
        life_durations[i+1] = 0
        visited_tiles[i+1] = []

        # Multi-agent based reproduction
        #multi_reproduction(i+1, parent, parent2)
        model_dict[i+1] = model_dict[parent]

        # Choosing the mutation 'intensity'
        # TODO: Try continuos mutation rate

        if life_durations[parent] > 60:
          mutate(i+1, 0.01)
        elif life_durations[parent] > 30:
          mutate(i+1, 0.05)
        else:
          mutate(i+1,0.1)

        #mutate(i+1, 0.02)
        # TODO: third mutation alternative - multiple parents

  # Calculate average lifetime of all agents every 20 steps
  avg_lifetime[i] = calculate_avg_lifetime()

  # The main loop
  for i in range(env.config.PLAYER_N):

    # Check if agents are alive, and if someone dies ignore their action
    if i+1 in env.realm.players.entities and i+1 in obs:
      #Increment life duration of each agent
      life_durations[i+1] += 1

      #Put the current pos of an agent into the visited positions dictionary
      visited_tiles[i+1].append(env.realm.players.entities[i+1].pos)


      all_visited.append(env.realm.players.entities[i+1].pos)

      input = get_input(obs[i+1]['Tile'], obs[i+1]['Entity'])
      output = model_dict[i+1][0](input)
      output_attack = model_dict[i+1][1](input)

      ### action_list.append(output)
      actions[i+1] = {"Move":{"Direction":1}, "Attack":{"Style":1,"Target":int(output_attack.argmax())}, "Use":{"InventoryItem":0}}

    else: actions[i+1] = {}

  # Run a step
  obs, rewards, dones, infos = env.step(actions)