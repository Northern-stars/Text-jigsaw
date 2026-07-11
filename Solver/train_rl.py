"""Train a DQN agent for the text jigsaw environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Optional

import torch

from Solver.agent.q_learning_agent import DQNJigsawAgent
from Solver.env.jigsaw_env import TextJigsawEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text jigsaw DQN agent.")
    parser.add_argument("--dataset-dir", default="Data/PuzzleData")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--category-weight", type=float, default=1.0)
    parser.add_argument("--done-reward", type=float, default=10.0)
    parser.add_argument("--reconstruct-text-by-line", action="store_true")
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--save-dir", default="Solver/checkpoints")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-feature-dim", type=int, default=256)
    parser.add_argument("--text-feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--text-num-heads", type=int, default=8)
    parser.add_argument("--text-num-layers", type=int, default=2)
    parser.add_argument("--text-max-length", type=int, default=512)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    env = TextJigsawEnv(
        dataset_dir=args.dataset_dir,
        pairwise_weight=args.pairwise_weight,
        category_weight=args.category_weight,
        done_reward=args.done_reward,
        max_steps=args.max_steps,
        reconstruct_text_by_line=args.reconstruct_text_by_line,
        seed=args.seed,
    )
    agent = DQNJigsawAgent(
        device=args.device,
        image_size=args.image_size,
        image_feature_dim=args.image_feature_dim,
        text_feature_dim=args.text_feature_dim,
        hidden_dim=args.hidden_dim,
        text_num_heads=args.text_num_heads,
        text_num_layers=args.text_num_layers,
        text_max_length=args.text_max_length,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_interval=args.target_update_interval,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(1, args.epochs + 1):
        metrics = _run_epoch(env, agent, args.episodes_per_epoch)
        _print_epoch_metrics(epoch, args.epochs, metrics, agent.epsilon)

        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(save_dir, epoch, args, agent)

    _save_checkpoint(save_dir, args.epochs, args, agent, name="last.pt")


def _run_epoch(
    env: TextJigsawEnv,
    agent: DQNJigsawAgent,
    episodes_per_epoch: int,
) -> dict[str, float]:
    episode_rewards = []
    episode_steps = []
    solved = []
    pairwise_rewards = []
    category_rewards = []
    losses = []

    for _ in range(episodes_per_epoch):
        obs = env.reset()
        total_reward = 0.0
        last_info: Optional[dict] = None

        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            loss = agent.optimize()

            if loss is not None:
                losses.append(loss)

            total_reward += reward
            obs = next_obs
            last_info = info

            if done or truncated:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(env.step_count)
        solved.append(1.0 if done else 0.0)
        if last_info is not None:
            pairwise_rewards.append(float(last_info["pairwise_reward"]))
            category_rewards.append(float(last_info["category_reward"]))

    return {
        "avg_reward": _safe_mean(episode_rewards),
        "avg_steps": _safe_mean(episode_steps),
        "solved_rate": _safe_mean(solved),
        "avg_pairwise_reward": _safe_mean(pairwise_rewards),
        "avg_category_reward": _safe_mean(category_rewards),
        "avg_loss": _safe_mean(losses),
    }


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    metrics: dict[str, float],
    epsilon: float,
) -> None:
    print(
        " ".join(
            [
                f"epoch={epoch}/{total_epochs}",
                f"avg_reward={metrics['avg_reward']:.4f}",
                f"avg_steps={metrics['avg_steps']:.2f}",
                f"solved_rate={metrics['solved_rate']:.4f}",
                f"pairwise={metrics['avg_pairwise_reward']:.4f}",
                f"category={metrics['avg_category_reward']:.4f}",
                f"epsilon={epsilon:.4f}",
                f"loss={metrics['avg_loss']:.6f}",
            ]
        )
    )


def _save_checkpoint(
    save_dir: Path,
    epoch: int,
    args: argparse.Namespace,
    agent: DQNJigsawAgent,
    name: Optional[str] = None,
) -> None:
    checkpoint_name = name or f"epoch_{epoch:04d}.pt"
    checkpoint_path = save_dir / checkpoint_name
    state = agent.state_dict()
    state.update(
        {
            "epoch": epoch,
            "config": vars(args),
        }
    )
    torch.save(state, checkpoint_path)


if __name__ == "__main__":
    main()
