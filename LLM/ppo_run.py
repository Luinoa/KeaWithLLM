# ppo_run.py
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import virtual_home
from policy import LLMAgent
from ppo_env import EnvRunner
from ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    # (Include all your argument definitions here)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    # (Include other algorithm and environment arguments as before)
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--policy-learning-rate", type=float, default=1e-6,
                        help="the learning rate of the optimizer")
    parser.add_argument("--value-learning-rate", type=float, default=3e-5,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=32,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--policy-num-minibatches", type=int, default=32,
                        help="the number of mini-batches")
    parser.add_argument("--value-num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=1,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--gradient-checkpointing-steps", type=int, default=8,
                        help="The number of steps for gradient checkpointing")
    parser.add_argument("--critic-warm-up-steps", type=int, default=5000,
                        help="The number of time steps to warm up critic")
    parser.add_argument("--env-id", type=str, default="VirtualHome-v1",
                        help="Domain name")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Whether to print debug information and render")
    parser.add_argument("--load-8bit", type=bool, default=False,
                        help="Whether to convert model to 8bits")
    parser.add_argument("--save-path", type=str, default="saved_models",
                        help="The path to save the checkpoint")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="The interval for saving model for certain num_updates")
    parser.add_argument("--resume", type=bool, default=False,
                        help="Whether to resume from previous checkpoint")
    parser.add_argument("--load-path", type=str, default="saved_models",
                        help="The path to load the checkpoint")
    parser.add_argument("--record-path", type=str, default="llm5_runs",
                        help="The path to save the tensorboard results")
    parser.add_argument("--normalization-mode", type=str, default="token",
                        help="The normalization mode for dealing with token logits")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.policy_minibatch_size = int(args.batch_size // args.policy_num_minibatches)
    args.value_minibatch_size = int(args.batch_size // args.value_num_minibatches)
    return args


if __name__ == "__main__":
    args = parse_args()
    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time_str}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"{args.record_path}/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env_params = {"seed": args.seed, "debug": args.debug}

    # Create the environment runner from ppo_env.py (fully decoupled)
    env_runner = EnvRunner(args, run_name, env_params, device)

    # Initialize the policy from policy.py
    if args.resume:
        agent = LLMAgent(normalization_mode=args.normalization_mode, load_path=args.load_path, load_8bit=args.load_8bit)
    else:
        agent = LLMAgent(normalization_mode=args.normalization_mode, load_8bit=args.load_8bit)

    # Storage setup for PPO
    obs = torch.zeros((args.num_steps, args.num_envs) + env_runner.envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + env_runner.envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    steps = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    pre_global_step = 0
    start_time = time.time()
    next_obs = env_runner.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    num_critic_warm_up_updates = args.critic_warm_up_steps // args.batch_size

    is_warmup = True
    trainer = PPOTrainer(agent, args, device, writer)

    for update in range(1, num_updates + 1 + num_critic_warm_up_updates):
        if is_warmup and update > num_critic_warm_up_updates:
            is_warmup = False

        # Anneal learning rates if needed (skip during warmup)
        if args.anneal_lr and not is_warmup:
            PPOTrainer.reset_optimizers(args, update)

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action selection
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step through environment via EnvRunner
            next_obs, reward, next_done, info = env_runner.step(action)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            for item in info:
                if "episode" in item.keys():
                    print(
                        f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # Compute bootstrap value and advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        experiences = {
            'obs': obs,
            'actions': actions,
            'logprobs': logprobs,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'next_obs': next_obs,
            'next_done': next_done,
        }

        stats = trainer.update(experiences, global_step, is_warmup)
        print(f"Update {update}, global_step: {global_step}, stats: {stats}")

        # Use trainer's method to log optimizer info
        optim_info = trainer.get_optimizer_info()
        writer.add_scalar("charts/policy_learning_rate", optim_info["policy_lr"], global_step)
        writer.add_scalar("charts/value_learning_rate", optim_info["value_lr"], global_step)
        writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

        if global_step // 10000 != pre_global_step // 10000:
            agent.save(global_step // 10000, f"{args.record_path}/{run_name}/{args.save_path}")
        pre_global_step = global_step

    agent.save(global_step // 10000 + 1, f"{args.record_path}/{run_name}/{args.save_path}")
    env_runner.close()
    writer.close()