from dataset import OfflineDataset
from torch.utils.data import DataLoader
import os
from utils import seed_everything
from env import make_env
import agents as Agents
from omegaconf import DictConfig
import hydra

@hydra.main(config_name="config2", version_base=None, config_path="config")
def main(cfg: DictConfig):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = cfg.util.gpu

    seed_everything(seed=cfg.util.seed)
    env = make_env(cfg.util.env_name)

    print("create dataset")
    train_dataset = OfflineDataset(data_dir=cfg.util.data_dir, log_file=cfg.util.log_file)
    print("number of data: ", len(train_dataset))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.util.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    print("get agent")
    agent_ = getattr(Agents, cfg.algo.algo)
    agent = agent_(
        gamma=cfg.env.gamma,
        epsilon=cfg.env.epsilon,
        lr=cfg.algo.lr,
        eps=cfg.algo.eps,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=cfg.algo.mem_size,
        eps_min=cfg.algo.eps_min,
        batch_size=cfg.algo.batch_size,
        replace=cfg.algo.replace,
        eps_dec=cfg.algo.eps_dec,
        chkpt_dir=cfg.util.chkpt_dir,
        algo=cfg.algo.algo,
        env_name=cfg.env.env_name,
        n_norm_groups=cfg.algo.n_norm_groups,
        dataloader=train_dataloader,
        alpha=cfg.algo.alpha,
        v_min=cfg.algo.v_min,
        v_max=cfg.algo.v_max,
        atoms=cfg.algo.atoms,
    )

    print("training")
    for i in range(cfg.algo.epochs):
        print("learn")
        train_loss = agent.learn()
        print(f"Epoch: {i+1} | " f"train_loss: {train_loss:.4f} |")

    
if __name__ == "__main__":
    main()

