import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.multiprocessing as _mp
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical

from env import MultipleEnvironments
from eval import eval
from model import PPO

os.environ['OMP_NUM_THREADS'] = '1'


def get_args():
    desc = "Implementation of model described in the paper: Proximal Policy Optimization Algorithms"
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--env_id", type=str, default='VSS3v3-v0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=9375)
    parser.add_argument("--num_episodes", type=int, default=2134)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200,
                        help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str,
                        default="tensorboard/ppo")
    parser.add_argument("--saved_path", type=str,
                        default="trained_models")
    parser.add_argument("--cuda", default=False,
                        action="store_true")
    parser.add_argument("--render", default=False,
                        action="store_true")
    args = parser.parse_args()
    return args


def main(opt):
    wandb.init(name=opt.env_id+'-PPO',
                   project='RC-Reinforcement',
                   dir='./logs')
    DEVICE = torch.device('cuda') if opt.cuda else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    # mp = _mp.get_context("fork")
    envs = MultipleEnvironments(opt.env_id, opt.num_processes)
    model = PPO(envs.num_states, envs.num_actions).to(DEVICE)
    # model.share_memory()
    # process = mp.Process(target=eval,
    #                      args=(opt, model, envs.num_states,
    #                            envs.num_actions, DEVICE))
    # process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.FloatTensor(curr_states).cpu()
    curr_episode = 0
    for _ in range(opt.num_episodes):
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(model.state_dict(), "{}/{}/{}".format(opt.saved_path,
                                                           opt.env_id,
                                                           curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        infos = []
        then = time.time()
        for _ in range(opt.num_steps):
            states.append(curr_states)
            actor_out, value = model(curr_states.to(DEVICE))
            action, old_log_policy = actor_out
            old_log_policy = old_log_policy.cpu()
            value = value.cpu()
            values.append(value)
            actions.append(action.cpu())
            old_log_policies.append(old_log_policy)
            [agent_conn.send(("step", act)) for agent_conn,
                act in zip(envs.agent_conns, action.detach().cpu())]
            del action
            state, reward, done, info = zip(
                *[agent_conn.recv() for agent_conn in envs.agent_conns])
            state = torch.FloatTensor(state)
            reward = torch.FloatTensor(reward)
            done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            curr_states = state

        rew_avg = torch.cat(rewards).sum()/opt.num_processes
        rew_avg = rew_avg.item()
        print('Average final reward per env:', rew_avg)
        fps = opt.num_steps*opt.num_processes/(time.time() - then)
        then = time.time()
        print('FPS:', fps)
        _, next_value, = model(curr_states.to(DEVICE))
        next_value = next_value.squeeze().cpu()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).squeeze().detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward \
                + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_steps * opt.num_processes)
            for j in range(opt.batch_size):
                total_steps = opt.num_steps * opt.num_processes
                num_batches = total_steps//opt.batch_size
                batch_indices = indice[j*num_batches:(j + 1)*num_batches]
                actor_outputs, value = model(states[batch_indices].to(DEVICE))
                actions, new_log_policy = actor_outputs
                ratio = torch.exp(
                    new_log_policy - old_log_policies[batch_indices].to(DEVICE)
                )
                basic_policy_opt = ratio * advantages[batch_indices].to(DEVICE)
                ppo_opt = torch.clamp(ratio,
                                      1.0 - opt.epsilon,
                                      1.0 + opt.epsilon)
                ppo_opt = ppo_opt * advantages[batch_indices].to(DEVICE)
                actor_loss = - \
                    torch.mean(torch.min(basic_policy_opt, ppo_opt))
                critic_loss = F.smooth_l1_loss(R[batch_indices].to(DEVICE),
                                               value.squeeze())
                entropy_loss = torch.mean(-new_log_policy)
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        log_dict = {"rw/Avg_rew": rew_avg, "FPS": fps}
        log_dict['Loss/Actor'] = actor_loss
        log_dict['Loss/Critic'] = critic_loss
        log_dict['Loss/Entropy'] = entropy_loss
        log_dict['Loss/Total'] = total_loss
        wandb.log(log_dict)
        print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))
        print('--------------------------------\n')

    # process.close()


if __name__ == "__main__":
    opt = get_args()
    main(opt)
