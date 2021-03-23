import multiprocessing as mp

import gym
import rc_gym


def create_env(env_id):
    return gym.make(env_id)


class MultipleEnvironments:
    def __init__(self, env_id, num_envs):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])

        self.envs = [create_env(env_id) for _ in range(num_envs)]
        self.num_states = self.envs[0].observation_space.shape[0]
        self.num_actions = self.envs[0].action_space.shape[0]
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action.numpy()))
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError
