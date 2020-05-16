import numpy as np
import torch as T
from models import Actor, Critic
from utils import ExperienceBuffer, EnvWrapper, calculate_G
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import torch.distributions as dist
import wandb
import sys

wandb.init(project="actor_critic_continuous")

PATH = 'Reacher_Linux_0/Reacher.x86_64'

env = EnvWrapper(PATH)

args = {'obs_space': env.observation_space,
        'action_space': env.action_space,
        'n_hidden': 128,
        'n_steps': 10,
        'lr_actor': 1e-5,
        'lr_critic': 1e-4,
        'device': 'cuda:1' if T.cuda.is_available() else 'cpu',
        'gamma': .95,
        'entropy_beta': 0.001,
        'critic_weight': 1,
        'episodes': 500,
        }

actor = Actor(args['obs_space'], args['action_space'], args['n_hidden'], args['lr_actor'], args['device'])
critic = Critic(args['obs_space'], 1, args['n_hidden'], args['lr_critic'], args['device'])
exp = ExperienceBuffer(args['n_steps'], args['device'])

for episode in range(args['episodes']):
    step = 0
    stats = {
        'rewards': 0,
        'actor_loss': 0,
        'critic_loss': 0,
        'entropy_loss': 0,
        'loss': 0
    }

    done = False
    state = env.reset()
    while not done:
        actor.eval()
        mus, sigams = actor(state)
        norm_dist = dist.Normal(mus, sigams)
        actions = norm_dist.sample()
        actions = np.clip(actions.detach().cpu().numpy(), -1., 1.)
        next_state, reward, dones, _ = env.step(actions)
        if dones[0]:
            done = True
        stats['rewards'] += reward[0]
        exp.add(state, actions, reward)


        if (len(exp) > 1) and (step % args['n_steps'] == 0):
            actor.train()
            critic.train()

            states, actions, rewards = exp.draw()
            values = critic(states).squeeze()
            mus, sigmas = actor(states)
            rewards = calculate_G(rewards, values, done, args['gamma'])
            rewards = rewards.to(args['device'])

            norm_dist = dist.Normal(mus, sigmas)
            logprobs = norm_dist.log_prob(actions).squeeze().mean(-1)
            entropy = norm_dist.entropy()
            entropy_loss = (-1 * args['entropy_beta'] * entropy).sum()
            actor_loss = (-1 * logprobs * (rewards - values.detach())).sum() + entropy_loss
            critic_loss = T.pow(values - rewards, 2).sum()
            loss = actor_loss + critic_loss
            actor.optimizer.zero_grad()
            critic.optimizer.zero_grad()

            loss.backward()
            actor.optimizer.step()
            critic.optimizer.step()

            stats['actor_loss'] += actor_loss
            stats['critic_loss'] += critic_loss
            stats['entropy_loss'] += entropy_loss
            stats['loss'] += loss

            exp.clear()

        step += 1

    wandb.log(stats)
    print(f'{stats["rewards"]:.2f} rewards in episode {episode}(steps: {step})')
T.save(actor, 'actor.h5')
T.save(critic, 'critic.h5')