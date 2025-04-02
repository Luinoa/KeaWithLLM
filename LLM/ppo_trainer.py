# ppo_trainer.py
import torch
import numpy as np
import torch.nn as nn

class PPOTrainer:
    def __init__(self, agent, args, device, writer):
        self.agent = agent
        self.args = args
        self.device = device
        self.writer = writer

        self.policy_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.agent.actor.parameters()),
            lr=args.policy_learning_rate, eps=1e-5, weight_decay=0
        )
        self.value_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.agent.critic.parameters()),
            lr=args.value_learning_rate, eps=1e-5
        )

    def update(self, experiences, global_step, is_warmup):
        args = self.args  # shorthand
        # Unpack experiences: obs, actions, logprobs, rewards, dones, values, next_obs, next_done
        obs = experiences['obs']
        actions = experiences['actions']
        logprobs = experiences['logprobs']
        rewards = experiences['rewards']
        dones = experiences['dones']
        values = experiences['values']
        num_steps, num_envs = rewards.shape

        # Compute bootstrap value and advantages
        with torch.no_grad():
            next_obs = experiences['next_obs']
            next_done = experiences['next_done']
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # Flatten the batch
        batch_size = args.num_envs * num_steps
        b_obs = obs.reshape((batch_size,) + obs.shape[2:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((batch_size,) + actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        kl_explode = False
        policy_update_steps = 0
        total_approx_kl = torch.tensor(0.0, device=self.device)

        # ----- Value Network Update -----
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.value_minibatch_size):
            end = start + args.value_minibatch_size
            mb_inds = b_inds[start:end]
            newvalue = self.agent.get_value(b_obs[mb_inds]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            loss = v_loss * args.vf_coef
            self.value_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
            self.value_optimizer.step()

        # Skip policy updates if still in warm-up
        if is_warmup:
            return {"value_loss": v_loss.item(), "policy_loss": 0.0, "approx_kl": 0.0}

        # ----- Policy Network Update -----
        self.policy_optimizer.zero_grad()
        for start in range(0, args.batch_size, args.policy_minibatch_size):
            # Handle gradient checkpointing (accumulate gradients)
            if policy_update_steps % args.gradient_checkpointing_steps == 0:
                total_approx_kl = torch.tensor(0.0, device=self.device)
            policy_update_steps += 1
            end = start + args.policy_minibatch_size
            mb_inds = b_inds[start:end]
            # Pass is_warmup flag to agent.get_action_and_value as needed.
            _, newlogprob, entropy, _ = self.agent.get_action_and_value(
                b_obs[mb_inds], b_actions.long()[mb_inds], is_warmup, return_value=False
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                total_approx_kl += approx_kl / args.gradient_checkpointing_steps
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss
            loss /= args.gradient_checkpointing_steps

            loss.backward()

            if policy_update_steps % args.gradient_checkpointing_steps == 0:
                if args.target_kl is not None and total_approx_kl > args.target_kl:
                    self.policy_optimizer.zero_grad()
                    kl_explode = True
                    policy_update_steps -= args.gradient_checkpointing_steps
                    break

                nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad()

        # Logging metrics
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/total_approx_kl", total_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/policy_update_times", policy_update_steps // args.gradient_checkpointing_steps, global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)

        return {"value_loss": v_loss.item(), "policy_loss": pg_loss.item(), "approx_kl": approx_kl.item()}

    def reset_optimizers(self, args, update):
        """ warm up the learning rate of the optimizers. """
        num_updates = args.total_timesteps // args.batch_size
        num_critic_warm_up_updates = args.critic_warm_up_steps // args.batch_size
        frac = 1.0 - (update - 1.0 - num_critic_warm_up_updates) / num_updates
        self.policy_optimizer.param_groups[0]["lr"] = frac * args.policy_learning_rate
        self.value_optimizer.param_groups[0]["lr"] = frac * args.value_learning_rate

    def get_optimizer_info(self):
        """Return optimizer learning rates for logging purposes."""
        info = {
            "policy_lr": self.policy_optimizer.param_groups[0]["lr"],
            "value_lr": self.value_optimizer.param_groups[0]["lr"],
        }
        return info