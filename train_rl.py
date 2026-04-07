import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from rl_env import HighwayObstacleEnv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = HighwayObstacleEnv(max_steps=120)
    env = Monitor(env)

    eval_env = Monitor(HighwayObstacleEnv(max_steps=120))

    # Save best model automatically during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/eval/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="models/checkpoints/",
        name_prefix="sac_highway",
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs/sac_highway_tensorboard/",
        learning_rate=3e-4,
        buffer_size=300000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    model.learn(
        total_timesteps=300000,
        callback=[eval_callback, checkpoint_callback],
    )

    model.save("models/sac_highway_obstacle")
    env.close()
    eval_env.close()

    print("Training complete. Model saved to models/sac_highway_obstacle")
    print("Best model saved to models/best_model")


if __name__ == "__main__":
    main()
