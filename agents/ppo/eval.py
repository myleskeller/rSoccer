"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
from env import create_env
from model import PPO


def eval(opt, global_model, state_size, action_size, device):
    torch.manual_seed(123)

    env = create_env(opt.env_id)
    local_model = PPO(state_size, action_size).to(device)
    local_model.eval()
    local_model.load_state_dict(global_model.state_dict(),
                                map_location=device)
    state = torch.from_numpy(env.reset()).to(device)
    done = False
    curr_step = 0
    epi_reward = 0
    while not done:
        curr_step += 1
        action, _ = local_model(state)
        action = action.detach().squeeze().numpy()
        state, reward, done, info = env.step(action)
        epi_reward += reward
        if opt.render:
            env.render()
        if done:
            state = env.reset()
        state = torch.from_numpy(state).to(device)


if __name__ == "__main__":

    print(eval)
