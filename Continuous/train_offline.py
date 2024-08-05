import gym
import csv
import datetime
from torch.utils.data import DataLoader
import os
import torch
from omegaconf import DictConfig
import hydra

from agents import ScaledQLAgent
from env import ImageEnv
from dataset import OfflineDataset


@hydra.main(config_name="config3", version_base=None, config_path="config")
def main(cfg: DictConfig):
    env = gym.make(cfg.env_id)
    env = ImageEnv(env)
    input_dims = (cfg.env.n_channels, cfg.env.img_size, cfg.env.img_size)
    agent = ScaledQLAgent(
        input_dims=input_dims,
        action_dim=env.action_space.shape[0],
        auto_alpha=cfg.algo.auto_alpha,
        alpha=cfg.algo.alpha,
        chkpt_dir=cfg.util.chkpt_dir,
        lr_actor=cfg.algo.lr_actor,
        lr_critic=cfg.algo.lr_critic,
        gamma=cfg.algo.gamma,
        hidden_size=cfg.algo.hidden_size,
        tau=cfg.algo.tau,
        batch_size=cfg.algo.batch_size,
        n_norm_groups=cfg.algo.n_norm_groups,
        cql_weight=cfg.algo.cql_weight,
        with_lagrange=cfg.algo.with_lagrange,
        target_action_gap=cfg.algo.target_action_gap,
        temparature=cfg.algo.temparature,
    )

    train_dataset = OfflineDataset(
        data_dir=cfg.util.memory_dir, log_file=cfg.util.log_file
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.algo.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    dateinfo = datetime.datetime.now()
    logfname = f"{cfg.util.log_dir}/{cfg.util.experiment_name}_{dateinfo.strftime('%Y%m%d%H%M%S')}.csv"
    with open(logfname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["steps", "avg_reward"])

    n_steps = 0
    last_reward = 0
    for epoch in range(cfg.algo.epochs):
        total_c1_loss_save = 0
        total_c2_loss_save = 0
        actor_loss_save = 0
        avg_q_values_save = 0
        for batch, data in enumerate(train_dataloader):
            total_c1_loss, total_c2_loss, actor_loss, avg_q_values = (
                agent.update_parameters(data)
            )
            total_c1_loss_save += total_c1_loss
            total_c2_loss_save += total_c2_loss
            actor_loss_save += actor_loss
            avg_q_values_save += avg_q_values
            n_steps += 1
            if n_steps % 2000 == 0:
                print("steps", n_steps)

        total_c1_loss_save /= len(train_dataloader)
        total_c2_loss_save /= len(train_dataloader)
        actor_loss_save /= len(train_dataloader)
        avg_q_values_save /= len(train_dataloader)
        print(
            "epoch:",
            epoch,
            "c1_loss %.1f" % total_c1_loss_save,
            "c2_loss %.1f" % total_c2_loss_save,
            "actor_loss %.1f" % actor_loss_save,
            "avg_q_value %.1f" % avg_q_values_save,
        )

        if epoch % cfg.util.eval_interval == 0:
            avg_reward = 0.0
            for _ in range(cfg.util.eval_interval):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                while True:
                    with torch.no_grad():
                        action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, truncated, terminated, _ = env.step(
                        action
                    )
                    episode_reward += reward
                    state = next_state
                    if done or truncated or terminated:
                        break
                avg_reward += episode_reward
            avg_reward /= cfg.util.eval_interval

            print("Epoch: {}, Eval Avg. Reward: {:.0f}".format(epoch, avg_reward))

            with open(logfname, "a") as f:
                writer = csv.writer(f)
                writer.writerow([n_steps, avg_reward])

            if avg_reward >= last_reward:
                agent.save_models(epoch)
                last_reward = avg_reward


if __name__ == "__main__":
    main()
