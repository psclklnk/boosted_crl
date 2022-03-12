import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, two_layers=False, **kwargs):
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape
        self._n_games = len(self._n_input)

        self._h1 = nn.Linear(self._n_input[0], n_features)
        if two_layers:
            self._h2 = nn.Linear(n_features, n_features)
        else:
            self._h2 = None
        self._q = nn.Linear(n_features, self._n_output[0])

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        if two_layers:
            nn.init.xavier_uniform_(self._h2.weight,
                                    gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._q.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        state = state.float()
        h = F.relu(self._h1(state))
        if self._h2 is not None:
            h = F.relu(self._h2(h))
        q = self._q(h)

        if action is not None:
            action = action.long()
            q = torch.squeeze(q.gather(1, action))

        return q
