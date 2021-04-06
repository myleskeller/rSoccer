import os
import shutil
import time
import traceback

import numpy as np
import ptan
import torch
import torch.autograd as autograd
from lib import common, ddpg_model
from ptan.experience import ExperienceSourceFirstLast
from tensorboardX import SummaryWriter

gradMax = 0
gradAvg = 0
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + '_best.pth')


def inspectGrads(grad):
    global gradMax, gradAvg
    maxg = grad.max()
    maxg = max(-grad.min(), maxg)
    # print("**** MAX GRAD: %.5f" % maxg + " OLD: %.5f" % gradMax + " AVG: %.5f" % gradAvg + " ****")
    if maxg > gradMax:
        print("**** NEW MAX GRAD: %.5f" % maxg + " OLD: %.5f" %
              gradMax + " AVG: %.5f" % gradAvg + " ****")
        gradMax = maxg
    gradAvg = .1*maxg + .9*gradAvg


def avg_rewards(exp_buffer, total):

    if total > len(exp_buffer):
        total = len(exp_buffer)

    count = 0
    reward = 0
    pos = exp_buffer.pos
    while count < total:
        reward += exp_buffer.buffer[pos].reward
        pos -= 1
        if pos < 0:
            pos = len(exp_buffer)-1
        count += 1

    return reward


def calc_loss_ddpg_critic(batch, crt_net, tgt_act_net, tgt_crt_net, gamma, cuda=False, cuda_async=False, per=False, mem_w=None):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    mem_loss = None

    states_v = torch.tensor(states, dtype=torch.float32)
    next_states_v = torch.tensor(next_states, dtype=torch.float32)
    actions_v = torch.tensor(actions, dtype=torch.float32)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    done_mask = torch.BoolTensor(dones)

    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)

    # critic
    q_v = crt_net(states_v, actions_v)
    last_act_v = tgt_act_net(next_states_v)
    q_last_v = tgt_crt_net(next_states_v, last_act_v)
    q_last_v[done_mask] = 0.0
    q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * gamma
    critic_loss_v = (q_v - q_ref_v.detach()).pow(2)
    if per:
        mem_w = Variable(torch.FloatTensor(mem_w))
        critic_loss_v = critic_loss_v * mem_w
        mem_loss = critic_loss_v
    critic_loss_v = critic_loss_v.mean()
    return critic_loss_v, mem_loss


def calc_loss_ddpg_actor(batch, act_net, crt_net, cuda=False, cuda_async=False):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = torch.tensor(states, dtype=torch.float32)
    next_states_v = torch.tensor(next_states, dtype=torch.float32)
    actions_v = torch.tensor(actions, dtype=torch.float32)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    done_mask = torch.BoolTensor(dones)

    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)

    # actor
    cur_actions_v = act_net(states_v)
    actor_loss_v = -crt_net(states_v, cur_actions_v)
    actor_loss_v = actor_loss_v.mean()
    return actor_loss_v


def create_actor_model(model_params, state_shape, action_shape, device):
    act_net = ddpg_model.DDPG_MODELS_ACTOR[model_params['act_type']](model_params['state_shape'].shape[0],
                                                                     model_params['action_shape'].shape[0]).to(device)
    return act_net


def load_actor_model(net, checkpoint):
    net.load_state_dict(checkpoint['state_dict_act'])

    return net


def play(params, net, device, exp_queue, agent_env, test, writer, collected_samples, finish_event):

    try:
        agentAtk = ddpg_model.AgentDDPG(net[0], device=device[0],
                                     ou_teta=params['ou_teta'],
                                     ou_sigma=params['ou_sigma'])
        
        agentGk = ddpg_model.AgentDDPG(net[1], device=device[1],
                                     ou_teta=params['ou_teta'],
                                     ou_sigma=params['ou_sigma'])



        print(f"Started from sample {collected_samples.value}.")
        state = agent_env.reset()
        matches_played = 0
        epi_reward_atk = 0
        epi_reward_gk = 0
        then = time.time()
        eval_freq_matches = params['eval_freq_matches']
        evaluation = False
        steps = 0

        while not finish_event.is_set():
            action_atk = agentAtk(state, steps)
            action_gk = agentGk(state, steps)
            next_state, reward, done, info = agent_env.step(action)
            steps += 1
            epi_reward_atk += reward[0]
            epi_reward_gk += reward[1]
            # agent_env.render()
            next_state = next_state if not done else None
            exp_atk = ptan.experience.ExperienceFirstLast(state, action_atk,
                                                      reward[0], next_state)
            exp_gk = ptan.experience.ExperienceFirstLast(state, action_gk,
                                                      reward[1], next_state)
            state = next_state
            if not test and not evaluation:
                exp_queue.put({'exp_atk': exp_atk, 'exp_gk': exp_gk})
            elif test:
                agent_env.render("human")

            if done:
                fps = steps/(time.time() - then)
                then = time.time()

                writer.add_scalar("rw/total_atk", epi_reward_atk, matches_played)
                writer.add_scalar("rw/total_gk", epi_reward_gk, matches_played)
                writer.add_scalar("rw/steps_ep", steps, matches_played)
                writer.add_scalar("rw/goal_score",
                                  info['goal_score'],
                                  matches_played)
                writer.add_scalar("rw/move_atk", info['move_atk'], matches_played)
                # writer.add_scalar("rw/move_gk", info['move_gk'], matches_played)
                writer.add_scalar("rw/move_y_gk", info['move_y_gk'], matches_played)
                writer.add_scalar(
                    "rw/ball_grad_atk", info['ball_grad_atk'], matches_played)
                writer.add_scalar("rw/energy_atk", info['energy_atk'], matches_played)
                writer.add_scalar("rw/goals_blue",
                                  info['goals_blue'],
                                  matches_played)
                writer.add_scalar("rw/goals_yellow",
                                  info['goals_yellow'],
                                  matches_played)
                writer.add_scalar("rw/defense_gk",
                                  info['defense_gk'],
                                  matches_played)
                writer.add_scalar("rw/distance_own_goal_bar_gk",
                                  info['distance_own_goal_bar_gk'],
                                  matches_played)
                writer.add_scalar("rw/ball_leave_area_gk",
                                  info['ball_leave_area_gk'],
                                  matches_played)
                

                print(f'<======Match {matches_played}======>')
                print(f'-------Reward:', epi_reward)
                print(f'-------FPS:', fps)
                print(f'<==================================>\n')
                epi_reward_atk = 0
                epi_reward_gk = 0
                steps = 0
                matches_played += 1
                state = agent_env.reset()
                agentAtk.ou_noise.reset()
                agentGk.ou_noise.reset()

                if not test and evaluation:  # evaluation just finished
                    writer.add_scalar("eval/rw_atk", epi_reward_atk, matches_played)
                    writer.add_scalar("eval/rw_gk", epi_reward_gk, matches_played)
                    print("evaluation finished")

                evaluation = matches_played % eval_freq_matches == 0

                if not test and evaluation:  # evaluation just started
                    print("Evaluation started")

            collected_samples.value += 1

    except KeyboardInterrupt:
        print("...Agent Finishing...")
        finish_event.set()

    except Exception:
        print("!!! Exception caught on agent!!!")
        print(traceback.format_exc())

    finally:
        if not finish_event.is_set():
            print("Agent set finish flag.")
            finish_event.set()

    agent_env.close()
    print("Agent finished.")


def train(model_params, act_net, device,
          exp_queue, finish_event, checkpoint=None):

    try:
        data_path = model_params['data_path']
        run_name_atk = model_params['run_name']['run_name_atk']
        run_name_gk = model_params['run_name']['run_name_gk']

        exp_buffer_atk = common.PersistentExperienceReplayBuffer(experience_source=None,
                                                             buffer_size=model_params['replay_size']) if \
            not model_params['per'] else common.PersistentExperiencePrioritizedReplayBuffer(experience_source=None,
                                                                                            buffer_size=model_params[
                                                                                                'replay_size'],
                                                                                            alpha=model_params['per_alpha'],
                                                                                            beta=model_params['per_beta'])
        exp_buffer_atk.set_state_action_format(
            state_format=model_params['state_format'], action_format=model_params['action_format'])

        exp_buffer_gk = common.PersistentExperienceReplayBuffer(experience_source=None,
                                                             buffer_size=model_params['replay_size']) if \
            not model_params['per'] else common.PersistentExperiencePrioritizedReplayBuffer(experience_source=None,
                                                                                            buffer_size=model_params[
                                                                                                'replay_size'],
                                                                                            alpha=model_params['per_alpha'],
                                                                                            beta=model_params['per_beta'])
        exp_buffer_gk.set_state_action_format(
            state_format=model_params['state_format'], action_format=model_params['action_format'])

        crt_net_atk = ddpg_model.DDPG_MODELS_CRITIC[model_params['crt_type']](model_params['state_shape'].shape[0],
                                                                          model_params['action_shape'].shape[0]).to(device['device_atk'])
        optimizer_act_atk = torch.optim.Adam(
            act_net['act_net_atk'].parameters(), lr=model_params['learning_rate'])
        optimizer_crt_atk = torch.optim.Adam(
            crt_net_atk.parameters(), lr=model_params['learning_rate'])
        tgt_act_net_atk = ptan.agent.TargetNet(act_net['act_net_atk'])
        tgt_crt_net_atk = ptan.agent.TargetNet(crt_net_atk)
        
        crt_net_gk = ddpg_model.DDPG_MODELS_CRITIC[model_params['crt_type']](model_params['state_shape'].shape[0],
                                                                          model_params['action_shape'].shape[0]).to(device['device_gk'])
        optimizer_act_gk = torch.optim.Adam(
            act_net['act_net_gk'].parameters(), lr=model_params['learning_rate'])
        optimizer_crt_gk = torch.optim.Adam(
            crt_net_gk.parameters(), lr=model_params['learning_rate'])
        tgt_act_net_gk = ptan.agent.TargetNet(act_net['act_net_gk'])
        tgt_crt_net_gk = ptan.agent.TargetNet(crt_net_gk)

        act_net['act_net_atk'].train(True)
        crt_net_atk.train(True)
        tgt_act_net_atk.target_model.train(True)
        tgt_crt_net_atk.target_model.train(True)

        act_net['act_net_gk'].train(True)
        crt_net_gk.train(True)
        tgt_act_net_gk.target_model.train(True)
        tgt_crt_net_gk.target_model.train(True)

        collected_samples = 0
        processed_samples = 0
        # best_reward = (-np.inf, -np.inf)
        best_reward_atk = -np.inf
        best_reward_gk = -np.inf

        if checkpoint is not None:
            if 'state_dict_crt' in checkpoint:
                if 'collected_samples' in checkpoint:
                    collected_samples = checkpoint['collected_samples']

                if 'processed_samples' in checkpoint:
                    processed_samples = checkpoint['processed_samples']

                reward_avg = checkpoint['reward']
                best_reward_atk = reward_avg['reward_atk']
                best_reward_gk = reward_avg['reward_gk']
                crt_net_atk.load_state_dict(checkpoint['state_dict_crt']['state_dict_crt_atk'])
                crt_net_gk.load_state_dict(checkpoint['state_dict_crt']['state_dict_crt_gk'])

                tgt_act_net_atk.target_model.load_state_dict(
                    checkpoint['tgt_act_state_dict']['tgt_act_state_dict_atk'])
                tgt_act_net_gk.target_model.load_state_dict(
                    checkpoint['tgt_act_state_dict']['tgt_act_state_dict_gk'])
                    
                tgt_crt_net_atk.target_model.load_state_dict(
                    checkpoint['tgt_crt_state_dict']['tgt_crt_state_dict_atk'])
                tgt_crt_net_gk.target_model.load_state_dict(
                    checkpoint['tgt_crt_state_dict']['tgt_crt_state_dict_gk'])

                optimizer_act_atk.load_state_dict(checkpoint['optimizer_act']['optimizer_act_atk'])
                optimizer_crt_atk.load_state_dict(checkpoint['optimizer_crt']['optimizer_crt_atk'])
                optimizer_act_gk.load_state_dict(checkpoint['optimizer_act']['optimizer_act_gk'])
                optimizer_crt_gk.load_state_dict(checkpoint['optimizer_crt']['optimizer_crt_gk'])
                print("=> loaded checkpoint '%s' (collected samples: %d, processed_samples: %d, with reward (attacker: %f, goalkeeper: %f))" % (
                    run_name, collected_samples, processed_samples, reward_avg['reward_atk'], reward_avg['reward_gk']))

            if 'exp' in checkpoint:  # load experience buffer
                exp = checkpoint['exp']
                exp_atk = exp_gk = None
                load_atk = True
                load_gk = True
                if exp is None:
                    print("Looking for default exb file for the attacker")
                    exp_atk = data_path + "/buffer/" + run_name_atk + ".exb"
                    load_atk = os.path.isfile(exp_atk)
                    if not load_atk:
                        print('File not found:"%s" (nothing to resume)' % exp_atk)
                    
                    print("Looking for default exb file for the goalkeeper")
                    exp_gk = data_path + "/buffer/" + run_name_gk + ".exb"
                    load_gk = os.path.isfile(exp_gk)
                    if not load_gk:
                        print('File not found:"%s" (nothing to resume)' % exp_gk)
                else:
                    exp_atk = exp['exp_atk']
                    exp_gk = exp['exp_gk']

                if load_atk:
                    print("=> Loading attacker experiences from: " + exp_atk + "...")
                    exp_buffer_atk.load_exps_from_file(exp_atk)
                    print("%d experiences loaded" % (len(exp_buffer_atk)))

                if load_gk:
                    print("=> Loading experiences from: " + exp_gk + "...")
                    exp_buffer_gk.load_exps_from_file(exp_gk)
                    print("%d experiences loaded" % (len(exp_buffer_gk)))


        target_net_sync = model_params['target_net_sync']
        replay_initial = model_params['replay_initial']
        next_check_point = processed_samples + \
            model_params['save_model_frequency']
        next_net_sync = processed_samples + model_params['target_net_sync']
        queue_max_size = batch_size = model_params['batch_size']
        writer_path = model_params['writer_path']
        writer_atk = SummaryWriter(log_dir=writer_path+"/attacker/train")
        writer_gk = SummaryWriter(log_dir=writer_path+"/goalkeeper/train")
        tracker_atk = common.RewardTracker(writer_atk)
        tracker_gk = common.RewardTracker(writer_gk)

        actor_loss_atk = 0.0
        critic_loss_atk = 0.0
        last_loss_average_atk = 0.0
        
        actor_loss_gk = 0.0
        critic_loss_gk = 0.0
        last_loss_average_gk = 0.0

        # training loop:
        print("Training started.")
        while not finish_event.is_set():
            new_samples = 0

            # print("get qsize: %d" % size)
            rewards_gg = [{'reward_atk': 0, 'reward_gk': 0} for _ in range(0, max(1, int(queue_max_size)))]
            for i in range(0, max(1, int(queue_max_size))):
                exp = exp_queue.get()
                if exp is None:
                    break
                exp_buffer_atk._add(exp['exp_atk'])
                exp_buffer_gk._add(exp['exp_gk'])
                rewards_gg[i] = {'reward_atk': exp['exp_atk'].reward, 'reward_gk': exp['exp_gk'].reward}
                new_samples += 1

            if len(exp_buffer_atk) < replay_initial or len(exp_buffer_gk) < replay_initial:
                continue

            collected_samples += new_samples

            # training loop:
            while exp_queue.qsize() < queue_max_size/2:
                mem_w_atk = None
                mem_w_gk = None
                if not model_params['per']:
                    batch_atk = exp_buffer_atk.sample(batch_size)
                    batch_gk = exp_buffer_gk.sample(batch_size)
                else:
                    batch_atk, mem_idxs_atk, mem_w_atk = exp_buffer_atk.sample(batch_size)
                    batch_gk, mem_idxs_gk, mem_w_gk = exp_buffer_gk.sample(batch_size)
                optimizer_crt_atk.zero_grad()
                optimizer_act_atk.zero_grad()
                optimizer_crt_gk.zero_grad()
                optimizer_act_gk.zero_grad()

                crt_loss_v_atk, mem_loss_atk = calc_loss_ddpg_critic(batch_atk, crt_net_atk, tgt_act_net_atk.target_model, tgt_crt_net_atk.target_model, gamma=model_params['gamma'],
                                                             cuda=(device['device_atk'].type == "cuda"), cuda_async=True, per=model_params['per'], mem_w=mem_w_atk)
                crt_loss_v_atk.backward()
                optimizer_crt_atk.step()

                crt_loss_v_gk, mem_loss_gk = calc_loss_ddpg_critic(batch_gk, crt_net_gk, tgt_act_net_gk.target_model, tgt_crt_net_gk.target_model, gamma=model_params['gamma'],
                                                             cuda=(device['device_gk'].type == "cuda"), cuda_async=True, per=model_params['per'], mem_w=mem_w_gk)
                crt_loss_v_gk.backward()
                optimizer_crt_gk.step()

                if model_params['per']:
                    mem_loss_atk = mem_loss_atk.detach().cpu().numpy()[0]
                    exp_buffer_atk.update_priorities(mem_idxs_atk, mem_loss_atk)
                    mem_loss_gk = mem_loss_gk.detach().cpu().numpy()[0]
                    exp_buffer_gk.update_priorities(mem_idxs_gk, mem_loss_gk)

                act_loss_v_atk = calc_loss_ddpg_actor(
                    batch_atk, act_net['act_net_atk'], crt_net_atk, cuda=(device['device_atk'].type == "cuda"),
                    cuda_async=True)
                act_loss_v_atk.backward()
                optimizer_act_atk.step()

                act_loss_v_gk = calc_loss_ddpg_actor(
                    batch_gk, act_net['act_net_gk'], crt_net_gk, cuda=(device['device_gk'].type == "cuda"),
                    cuda_async=True)
                act_loss_v_gk.backward()
                optimizer_act_gk.step()

                processed_samples += batch_size
                critic_loss_atk += crt_loss_v_atk.item()
                actor_loss_atk += act_loss_v_atk.item()
                critic_loss_gk += crt_loss_v_gk.item()
                actor_loss_gk += act_loss_v_gk.item()

            # print("|\n")

            # soft sync

            if target_net_sync >= 1:
                if processed_samples >= next_net_sync:
                    next_net_sync = processed_samples + target_net_sync
                    tgt_act_net_atk.sync()
                    tgt_crt_net_atk.sync()
                    tgt_act_net_gk.sync()
                    tgt_crt_net_gk.sync()
            else:
                tgt_act_net_atk.alpha_sync(alpha=target_net_sync)  # 1 - 1e-3
                tgt_crt_net_atk.alpha_sync(alpha=target_net_sync)
                tgt_act_net_gk.alpha_sync(alpha=target_net_sync)  # 1 - 1e-3
                tgt_crt_net_gk.alpha_sync(alpha=target_net_sync)

            if processed_samples >= next_check_point:
                next_check_point = processed_samples + \
                    model_params['save_model_frequency']
                reward_avg = {'reward_atk': avg_rewards(exp_buffer_atk, 1000), 'reward_gk': avg_rewards(exp_buffer_gk, 1000)}

                if reward_avg['reward_atk'] > best_reward_atk:
                    best_reward_atk = reward_avg['reward_atk']
                    is_best_atk = True
                else:
                    is_best_atk = False

                if reward_avg['reward_gk'] > best_reward_gk:
                    best_reward_gk = reward_avg['reward_gk']
                    is_best_gk = True
                else:
                    is_best_gk = False

                try:
                    print("saving attacker checkpoint with %d/%d collected/processed samples with best reward %f..." %
                          (collected_samples, processed_samples, best_reward_atk))
                    save_checkpoint({
                        'model_type': 'ddpg',
                        'collected_samples': collected_samples,
                        'processed_samples': processed_samples,
                        'state_dict_act': act_net['act_net_atk'].state_dict(),
                        'state_dict_crt': crt_net_atk.state_dict(),
                        'tgt_act_state_dict': tgt_act_net_atk.target_model.state_dict(),
                        'tgt_crt_state_dict': tgt_crt_net_atk.target_model.state_dict(),
                        'reward': reward_avg['reward_atk'],
                        'optimizer_act': optimizer_act_atk.state_dict(),
                        'optimizer_crt': optimizer_crt_atk.state_dict(),
                    }, is_best_atk, "model/" + run_name_atk + ".pth")

                    if processed_samples > last_loss_average_atk:
                        actor_loss_atk = batch_size*actor_loss_atk / \
                            (processed_samples-last_loss_average_atk)
                        critic_loss_atk = batch_size*critic_loss_atk / \
                            (processed_samples-last_loss_average_atk)
                        print("avg_reward_atk:%.4f, avg_loss_atk:%f" %
                              (reward_avg['reward_atk'], actor_loss_atk))
                        tracker_atk.track_training(
                            processed_samples, reward_avg['reward_atk'], actor_loss_atk, critic_loss_atk)
                        actor_loss_atk = 0.0
                        critic_loss_atk = 0.0
                        last_loss_average_atk = processed_samples

                    exp_buffer_atk.sync_exps_to_file(
                        data_path + "/buffer/" + run_name_atk + ".exb")

                except Exception:
                    with open(run_name_atk + ".err", 'a') as errfile:
                        errfile.write("!!! Exception caught on training !!!")
                        errfile.write(traceback.format_exc())

                try:
                    print("saving goalkeeper checkpoint with %d/%d collected/processed samples with best reward %f..." %
                          (collected_samples, processed_samples, best_reward_gk))
                    save_checkpoint({
                        'model_type': 'ddpg',
                        'collected_samples': collected_samples,
                        'processed_samples': processed_samples,
                        'state_dict_act': act_net['act_net_gk'].state_dict(),
                        'state_dict_crt': crt_net_gk.state_dict(),
                        'tgt_act_state_dict': tgt_act_net_gk.target_model.state_dict(),
                        'tgt_crt_state_dict': tgt_crt_net_gk.target_model.state_dict(),
                        'reward': reward_avg['reward_gk'],
                        'optimizer_act': optimizer_act_gk.state_dict(),
                        'optimizer_crt': optimizer_crt_gk.state_dict(),
                    }, is_best_gk, "model/" + run_name_gk + ".pth")

                    if processed_samples > last_loss_average:
                        actor_loss_gk = batch_size*actor_loss_gk / \
                            (processed_samples-last_loss_average_gk)
                        critic_loss_gk = batch_size*critic_loss_gk / \
                            (processed_samples-last_loss_average_gk)
                        print("avg_reward_gk:%.4f, avg_loss_gk:%f" %
                              (reward_avg['reward_gk'], actor_loss_gk))
                        tracker_gk.track_training(
                            processed_samples, reward_avg['reward_gk'], actor_loss_gk, critic_loss_gk)
                        actor_loss_gk = 0.0
                        critic_loss_gk = 0.0
                        last_loss_average_gk = processed_samples

                    exp_buffer_gk.sync_exps_to_file(
                        data_path + "/buffer/" + run_name_gk + ".exb")

                except Exception:
                    with open(run_name_gk + ".err", 'a') as errfile:
                        errfile.write("!!! Exception caught on training !!!")
                        errfile.write(traceback.format_exc())

    except KeyboardInterrupt:
        print("...Training finishing...")

    except Exception:
        print("!!! Exception caught on training !!!")
        print(traceback.format_exc())
        with open(run_name_atk+".err", 'a') as errfile:
            errfile.write("!!! Exception caught on training !!!")
            errfile.write(traceback.format_exc())
            
        with open(run_name_gk+".err", 'a') as errfile:
            errfile.write("!!! Exception caught on training !!!")
            errfile.write(traceback.format_exc())

    finally:
        if not finish_event.is_set():
            print("Train process set finish flag.")
            finish_event.set()

        print("Training finished.")
