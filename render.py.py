import gym
import pandas as pd
from gym.wrappers.monitoring.video_recorder import VideoRecorder


env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)

data = []
video = VideoRecorder(env, "mountainCar.mp4")

env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    video.capture_frame()
    data.append(observation[0])
    df = pd.DataFrame(data, columns=["CarPosition"])
    df.to_csv('./carPosition.csv')

video.close()
env.close()