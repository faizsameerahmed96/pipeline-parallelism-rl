import gymnasium as gym
import numpy as np

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", step_trigger=lambda step: step % 50_000 == 0)
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)

        new_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
        env = gym.wrappers.TransformObservation(env, lambda obs:  obs.astype(np.float32) / 255.0, observation_space=new_space)
        
        return env

    return thunk