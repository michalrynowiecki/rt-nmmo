class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(hidden_size2, output_size)   # Second hidden layer to output layer
        self.tanh = nn.Tanh()  # Tanh activation function

    def forward(self, x):
        x = self.tanh(self.fc1(x))  # Pass through first hidden layer with tanh activation
        x = self.tanh(self.fc2(x))  # Pass through second hidden layer with tanh activation
        x = self.tanh(self.fc3(x))  # Pass through output layer with tanh activation
        return x
    
def initialize_networks(input_size, hidden_size1, hidden_size2, output_size, player_N):
    # Define the model
    input_size = 3775
    hidden_size1 = 225
    hidden_size2 = 75
    output_size = 5
    output_size_attack = player_N+1

    # Random weights with a FF network
    model_dict = {i+1: (FeedForwardNN(input_size, hidden_size1, hidden_size2, output_size), FeedForwardNN(input_size, hidden_size1, hidden_size2, output_size_attack))  for i in range(player_N)} # Dictionary of random models for each agent

    return model_dict

def get_actions():
    # Forward pass with a feed forward NN
    action_list = []
    action_list_attack = []

    for i in range(env.config.PLAYER_N):
        if (env.realm.players.corporeal[1].alive):
            # Get the observations
            input = get_input(obs[i+1]['Tile'], obs[i+1]['Entity'])
            # Get move actions
            output = model_dict[i+1][0](input)
            # Get attack actions (target, since agents only do melee combat)
            output_attack = model_dict[i+1][1](input)
            action_list.append(output.argmax())
            action_list_attack.append(output_attack.argmax())

    actions = {}
    for i in range(env.config.PLAYER_N):
        actions[i+1] = {"Move":{"Direction":1}, "Attack":{"Style":1,"Target":int(action_list_attack[i])}}

    return actions