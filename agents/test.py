import gym
import rc_gym

# Using penalty env
#env = gym.make('VSS3v3-v0')
env = gym.make('VSSMotionControl-v0')


# Run for 1 episode and print reward at the end
for i in range(1000):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
    print(reward)