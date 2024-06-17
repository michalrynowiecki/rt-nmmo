# Provide tile and entity observations to receive neural net input
def get_input(tile, entity):
  return torch.tensor(np.concatenate((tile.reshape(-1), entity.reshape(-1)), axis = None)).float()

# Functions for saving and loading neural network weights
def save_state(models_dictionary, save_path):
  for i in models_dictionary:
    torch.save(model_dict[i][0].state_dict(), f"{save_path}/agent_move_{i}")
    torch.save(model_dict[i][1].state_dict(), f"{save_path}/agent_attack_{i}")

def load_state(models_dictionary, load_path):
  for i in models_dictionary:
    model_dict[i][0].load_state_dict(torch.load(f"{load_path}/agent_move_{i}"))
    model_dict[i][1].load_state_dict(torch.load(f"{load_path}/agent_attack_{i}"))

#Checking for how long each of the agents has travelled during the course of its life
def get_distance_travelled(entities, spawn_positions, agent_number):
  x_2, y_2 = entities[agent_number].pos #current position coordinates
  x_1, y_1 = spawn_positions[agent_number] #spawn position coordinates

  dist_squared = (x_2 - x_1)**2 + (y_2 - y_1)**2
  if (dist_squared) != 0:
    return sqrt(dist_squared)
  else:
    return 0
#EXAMPLE: get_distance_travelled(env.realm.players.entities, env.game_state.spawn_pos, 3)

def mutate(player_num, alpha=0.01):
  # mutate movement network
  for param in model_dict[player_num][0].parameters():
    with torch.no_grad():
      param.add_(torch.randn(param.size()) * alpha)
  # mutate attack network
  for param in model_dict[player_num][1].parameters():
    with torch.no_grad():
      param.add_(torch.randn(param.size()) * alpha)

#fitness function
def fitness(player_number, environment):
  fitness_dict = {}

  # Calculate distance travelled for each agent
  for i in range(player_number):
    if i+1 in environment.realm.players.entities:
      fitness_dict[i+1] = get_distance_travelled(environment.realm.players.entities, environment.game_state.spawn_pos, i+1)

  # Get index of max value (the best agent number)
  best_agent = max(fitness_dict, key=fitness_dict.get)

  return fitness_dict

#Select a parent from top beta percent
def pick_best(beta):
  beta = ceil(beta*env.config.PLAYER_N)

  # Get beta top percent of both
  my_keys_long_runners = sorted(fitness(), key=fitness().get, reverse=True)[:beta]
  my_keys_long_livers = sorted(life_durations, key=life_durations.get, reverse=True)[:beta]
  bestest = list(set(my_keys_long_runners).intersection(my_keys_long_livers))

  # Pick the best from the intersetction of the longest living agents and furthest walking agents. If that is empty then pick one from the longest living agents.
  if bestest:
    parent = random.choice(bestest)
  else:
    parent = random.choice(my_keys_long_livers)


  return parent

# Combining two neural networks
def multi_reproduction(child, parent1, parent2):
  for layer in model_dict[child][0].state_dict().keys():
    l = random.randint(1, 2)
    # randomly pick whose parents layer the current layer should be
    match l:
      case 1:
        model_dict[child][0].state_dict()[layer] = model_dict[parent1][0].state_dict()[layer]
        break
      case 2:
        model_dict[child][0].state_dict()[layer] = model_dict[parent2][0].state_dict()[layer]
        break
  return

#Calculate current average lifetime of all players
def calculate_avg_lifetime(player_N, environment):
  sum = 0
  for i in range(player_N):
    if i+1 in environment.realm.players.entities and i+1 in obs:
      sum += environment.realm.players.entities[i+1].__dict__['time_alive'].val
  sum = sum/player_N
  return sum


