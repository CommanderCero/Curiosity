import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_feedforward(sizes, activation=nn.ReLU): 
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

def get_output_count(net: nn.Module, input_shape: tuple):
    with torch.no_grad():
        o = net(torch.zeros(1, *input_shape))
        return int(np.prod(o.size()))

class Conv2DEmbedding(nn.Module):
    def __init__(self, input_shape, embedding_size):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
        )
        
        input_size = get_output_count(self.conv_net, input_shape)
        self.out_net = create_feedforward([input_size, embedding_size])
        
    def forward(self, X):
        X = self.conv_net(X)
        X = torch.flatten(X, start_dim=1)
        X = self.out_net(X)
        return X
    
    

class OneHotForwardModel(nn.Module):
    '''
    Implements a forward model that predicts the next state using discrete actions as input.
    The actions are handled by using one-hot encoding.
    '''
    def __init__(self, state_predictor_net: nn.Module, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.state_predictor_net = state_predictor_net
    
    def forward(self, states, actions):
        '''
        Predicts the next state from the given (state, action) pairs.
        Assumptions:
            states: [batch_size, embedding_size]
            actions: [batch_size, 1]
        '''
        # [batch_size, 1, num_actions]
        one_hot_actions = F.one_hot(actions, num_classes=self.num_actions)
        # [batch_size, num_actions]
        one_hot_actions = one_hot_actions.squeeze(1)
        
        # [batch_size, embedding_size + num_actions]
        X = torch.cat([states, one_hot_actions], dim=1)
        # [batch_size, embedding_size]
        state_predictions = self.state_predictor_net(X)
        return state_predictions
    
class CategoricalActionPredictor(nn.Module):
    def __init__(self, logits_net: nn.Module):
        super().__init__()
        self.logits_net = logits_net
    
    def forward(self, states, next_states) -> torch.distributions.Distribution:
        '''
        Predicts the executed actions from the given (state, next_state) pairs.
        Assumptions:
            states: [batch_size, embedding_size]
            next_states: [batch_size, embedding_size]
        '''
        # [batch_size, embedding_size * 2]
        X = torch.concat([states, next_states], dim=1)
        # [batch_size, num_actions]
        logits = self.logits_net(X)
        return torch.distributions.Categorical(logits=logits)

