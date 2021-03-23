"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.actor = self.net = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, num_actions),
            nn.Tanh()
        )
        self.log_std = -0.5 * torch.ones(num_actions).float()
        self.std = torch.exp(self.log_std)
        self._initialize_weights()

    def to(self, *args):
        self = super().to(*args)
        self.log_std = self.log_std.to(*args)
        self.std = self.std.to(*args)
        return self

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(
                    module.weight, nn.init.calculate_gain('relu')
                )
                nn.init.constant_(module.bias, 0)

    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(torch.exp(log_std)+1e-8))
                          ** 2 + 2*log_std + np.log(2*np.pi))
        return torch.sum(pre_sum, axis=1)

    def forward(self, x):
        pred = self.actor(x)
        low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
        action = pred + torch.rand(pred.shape, device=pred.device) * self.std
        action = torch.clip(action, low, high)
        log = self.gaussian_likelihood(action, pred, self.log_std)
        actor_out = (action, log)
        return actor_out, self.critic(x)


if __name__ == "__main__":
    import torch
    ppo = PPO(3, 2)
    a = torch.ones((1, 3))
    action, value = ppo(a)
    print(action, value)
