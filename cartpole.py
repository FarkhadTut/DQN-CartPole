import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as T
import matplotlib 
import matplotlib.pyplot as plot
import random
import gym
import math
import sys
from collections import namedtuple
from PIL import Image
from operator import itemgetter
from itertools import count
from random import randint

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
print(is_ipython)

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Experience = namedtuple('Experience',('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
  def __init__(self, img_height, img_width, num_actions):
    super().__init__()
    self.conv1 = nn.Conv2d(1,16,8, stride=4, padding=1)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16,32,4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32,64,3)
    self.bn3 = nn.BatchNorm2d(64)
    
    self.fc1 = nn.Linear(in_features = 7569, out_features = 256)
    self.denseBn1 = nn.BatchNorm1d(256)
    self.fc2 = nn.Linear(in_features = 256, out_features = 128)
    self.denseBn2 = nn.BatchNorm1d(128)
    self.fc3 = nn.Linear(in_features = 128, out_features = 64)
    self.denseBn3 = nn.BatchNorm1d(64)
    self.out = nn.Linear(in_features = 64, out_features = num_actions)

    self.init_bias()


  def init_bias(self):
    nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
    nn.init.constant_(self.conv1.bias, 0)
    nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
    nn.init.constant_(self.conv2.bias, 0)
    nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
    nn.init.constant_(self.conv3.bias, 0)

  def forward(self, t):
    t = t.to(device)
    # t = self.bn1(self.conv1(t))
    # t = self.bn2(self.conv2(t))
    # t = self.bn3(self.conv3(t))
    # print(t.flatten(start_dim=1).shape)
    # sys.exit()
    # t = F.relu(self.conv1(t))
    # t = F.relu(self.conv2(t))
    # t = F.relu(self.conv3(t))

    t = t.flatten(start_dim=1)
    t = F.relu(self.fc1(t))
    t = F.relu(self.fc2(t))
    t = F.relu(self.fc3(t))


    # t = F.relu(self.fc1(t))

    t = F.relu(self.out(t))
    return t

def extract_tensors(experiences):
  batch = Experience(*zip(*experiences))

  t1 = torch.cat(batch.state)
  t2 = torch.cat(batch.action)
  t3 = torch.cat(batch.next_state)
  t4 = torch.cat(batch.reward)
  return (t1, t2, t3, t4)

class QValues():
  @staticmethod
  def get_current(policy_net, states, actions):
    return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

  # @staticmethod
  # def get_next(target_net, next_states):
  #   non_final_states_locations = next_states.flatten(start_dim=1).max(dim=1)[0].type(torch.bool)
  #   non_final_states = next_states[non_final_states_locations]
  #   batch_size = next_states.shape[0]
  #   values = torch.zeros(batch_size).to(device)
  #   values[non_final_states_locations] = target_net(non_final_states).max(dim=1)[0].detach()
  #   return values

  @staticmethod
  def get_next(target_net, next_states):
    final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
    non_final_state_locations = (final_state_locations == False)
    non_final_states = next_states[non_final_state_locations]
    batch_size = next_states.shape[0]
    values = torch.zeros(batch_size).to(device)
    values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
    return values



class Agent():
  def __init__(self, strategy, num_actions):
    self.current_step = 0
    self.num_actions = num_actions
    self.strategy = strategy

  def select_action(self, policy_net, target_net, state):
    rate = self.strategy.get_exploration_rate(self.current_step)
    self.current_step += 1
    

    if rate > random.random():
      action = random.randrange(self.num_actions)
      return torch.tensor([action]).to(device)
    else:
      with torch.no_grad():
        # if not NET_CHOOSER:
        #   policy_net.eval()
        #   action = (policy_net(state)/2 + target_net(state)/2).argmax(dim=1).to(device)
        #   policy_net.train()
        # else:
        #   target_net.eval()
        #   action = (policy_net(state)/2 + target_net(state)/2).argmax(dim=1).to(device)
        #   target_net.train()

        action = target_net(state).argmax(dim=1).to(device)
        return action

class EpsilonGreedyStrategy():
  def __init__(self, eps_start, eps_end, eps_decay, linear=False):
    self.start = eps_start
    self.end = eps_end
    self.decay = eps_decay
    self.rate = eps_start
    self.linear = linear

  def get_exploration_rate(self, current_step):
    if not self.linear:
      self.rate = self.end + (self.start - self.decay) * math.exp(-1.*current_step * self.decay)
    else:
      if self.rate > self.end:
        self.rate = self.start - current_step*self.decay
    return self.rate



class ReplayMemory():
  def __init__(self, capacity, priority_scale):
    self.capacity = capacity
    self.memory = []
    self.push_count = 0
    self.priority_scale = priority_scale
    self.sample_probs = []
    self.max_priority = 110.0
    self.priorities = np.array([])

  def push(self, experience):
    if len(self.memory) < self.capacity:
      self.memory.append(experience)
      self.priorities = np.append(self.priorities, self.max_priority)
    else:
      replace_idx = self.push_count % self.capacity
      self.memory[replace_idx] = experience
      self.priorities[replace_idx] = self.max_priority

    self.push_count += 1

  def refresh_priorities(self, errors, idx, OFFSET):
    errors = np.array(list(map(lambda x: (x.item()**2)**(0.5), errors)))
    self.priorities[idx] = errors + OFFSET

  def can_provide_sample(self, batch_size):
    return len(self.memory) >= batch_size

  def get_probs(self):
    scaled_priorities = self.priorities**self.priority_scale
    self.sample_probs = scaled_priorities/sum(scaled_priorities)*100000
    return self.sample_probs

  def sample(self, batch_size):
    sample_probs = self.get_probs()
    idx = random.choices(range(len(self.memory)), k=batch_size, weights=sample_probs)
    sampled = list(itemgetter(*idx)(self.memory))
    return sampled, idx

  def get_importance(self, idx):
    importance = 1/self.capacity - 1/self.sample_probs
    importance = importance/sum(importance)
    sampled_importance = importance[idx]  
    return sampled_importance



class EnvManager():
  def __init__(self, device):
    self.device = device
    self.env = gym.make('CartPole-v0').unwrapped
    self.env.reset()
    self.current_screen = None
    self.done = False
    self.last_state = None
    self.lives_total = 0
    self.num_current_frame = 0

  def reset(self):
    self.env.reset()
    self.current_screen = None

  def close(self):
    self.env.close()

  def render(self, mode="human"):
    return self.env.render(mode)

  def num_actions_available(self):
    return self.env.action_space.n


  def take_action(self, action):
    _, reward, self.done, info = self.env.step(action.item())
    return torch.tensor([reward], device = self.device), info

  def just_starting(self):
    return self.current_screen is None

  def get_terminal_black(self):
    black_screen = torch.zeros_like(self.get_processed_screen())
    return black_screen


  def get_state(self):
    if self.done:
      self.current_screen = None
      return self.get_terminal_black() 

    if self.just_starting():
      self.last_state = self.get_processed_screen()
      state = self.last_state
      self.current_screen = state

      return state

    s1 = self.last_state
    s2 = self.get_processed_screen()
    self.last_state = s2
    return s2 - s1

# 
  def get_screen_height(self):
    screen = self.get_processed_screen()
    return screen.shape[2]

  def get_screen_width(self):
    screen = self.get_processed_screen()
    return screen.shape[3]

  def get_processed_screen(self):
    screen = self.render("rgb_array")
    screen = self.gray(screen)
    screen = self.crop_screen(screen)
    return self.transform_screen_data(screen)
  
  def gray(self, screen):
    r, g, b = screen[:,:,0], screen[:,:,1], screen[:,:,2]
    gray_screen = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_screen

  def crop_screen(self, screen):
    # screen_height = screen.shape[0]
    # top = int(screen_height*0.16)
    # bottom = int(screen_height*0.925)
    # screen = screen[top:bottom, :]

    # screen_width = screen.shape[1]
    # left = int(screen_width*0.05)
    # right = int(screen_width*0.95)
    # screen = screen[:, left:right]

    screen_height = screen.shape[0]
    top = int(screen_height*0.4) #0.097
    bottom = int(screen_height*0.8) #0.93
    screen = screen[top:bottom, :]

    return screen

  def transform_screen_data(self, screen):
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.from_numpy(screen)

    resize = T.Compose([
      T.ToPILImage(),
      T.Resize((87,87)),
      T.ToTensor()
    ])

    return resize(screen).unsqueeze(0).to(device)


def plot(values, moving_avg_period, rate, init_episode, frames):
  plt.figure(2)
  plt.clf()
  plt.title('Training...')
  plt.xlabel('Episode')
  plt.ylabel('Duration')
  plt.plot(values)
  moving_avg = get_moving_average(moving_avg_period, values)
  plt.plot(moving_avg)
  plt.pause(0.01)
  print("Episode: ", init_episode + len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
  print("Explor. rate: ", rate)
  print("Frames: ", frames)
  print()
  plt.savefig("/home/fara-ubuntu/Documents/FARA/dqn/breakout_plot.png")
  if is_ipython: display.clear_output(wait=True)


def plot_loss(values, moving_avg_period):
  plt.figure(3)
  plt.clf()
  plt.title('LOSS')
  plt.xlabel('Episode')
  plt.ylabel('Loss')
  plt.plot(values)
  moving_avg = get_moving_average(moving_avg_period, values)
  plt.plot(moving_avg)
  plt.pause(0.01)

  if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
  values = torch.tensor(values, dtype=torch.float)
  if len(values) >= period:
    moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
    moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
    return moving_avg
  else:
    moving_avg = torch.zeros(len(values))
    return moving_avg


batch_size = 64
gamma = 0.998
eps_start = 1
eps_end = 0.05
eps_decay = 0.0005
target_update = 10
priority_update = 5
memory_size = 100000
lr = 0.001
num_episodes = 500000
# PRIORITIZATION
OFFSET = 10
PRIORITY_SCALE = 0
BETTA_P = 0
BETTA_T = 0
BETTA_END = 1
BETTA_STEP = 0.000
NET_CHOOSER = True

best_time = 0
current_time = 0
best_score = 0
losses = []

frames = 0

print(device)
em = EnvManager(device)
# checkpoint = torch.load("/home/fara-ubuntu/Documents/FARA/dqn/Atari_save/DQN_PARAM_BREAK.pt")
# eps_start = checkpoint['eps_start']
# init_episode = checkpoint['episode']
# frames = checkpoint['frames']
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay, linear=False)
agent = Agent(strategy, em.num_actions_available())
memory = ReplayMemory(memory_size, PRIORITY_SCALE)

policy_net = DQN(em.get_screen_height(), em.get_screen_width(), agent.num_actions).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width(), agent.num_actions).to(device)

# policy_net.load_state_dict(checkpoint['policy_net'])
# target_net.load_state_dict(checkpoint['target_net'])

poptimizer = optim.RMSprop(params=policy_net.parameters(), lr=lr)
toptimizer = optim.Adam(params=target_net.parameters(), lr=lr)

# poptimizer.load_state_dict(checkpoint['poptimizer'])

episode_durations = []
scores = []

policy_net.train()
target_net.eval()





# for episode in range(num_episodes):
#   em.reset()
#   state = em.get_state()
#   score = 0
#   if NET_CHOOSER:
#     policy_net.train()
#     target_net.eval()
#     NET_CHOOSER = not NET_CHOOSER
#   else:
#     policy_net.eval()
#     target_net.train()
#     NET_CHOOSER = not NET_CHOOSER

#   for timestep in count():
#     action = agent.select_action(policy_net, target_net, state)
#     reward = em.take_action(action)
#     score += reward
#     next_state = em.get_state()
#     # next_observation = em.get_state()
#     # next_state = state
#     # next_state[0][0] = next_state[0][1]
#     # next_state[0][1] = next_state[0][2]
#     # next_state[0][2] = next_state[0][3]
#     # next_state[0][3] = next_observation[0][0]
#     memory.push(Experience(state, action, next_state, reward))
#     state = next_state

#     if memory.can_provide_sample(batch_size):
#       experiences, idx = memory.sample(batch_size)
#       states, actions, next_states, rewards = extract_tensors(experiences)

#       if BETTA_P <= 1:
#         BETTA_P += BETTA_STEP
#       if BETTA_T <=1:
#         BETTA_T += BETTA_STEP
      
#       if NET_CHOOSER:
#         current_q_values = QValues.get_current(policy_net, states, actions)
#         next_q_values = QValues.get_next(target_net, next_states)
#         target_q_values = (next_q_values*gamma) + rewards
        
#         loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1), reduction='none')
#         loss = torch.transpose(loss, 0, 1).squeeze()
#         w = torch.from_numpy(memory.get_importance(idx)).to(device)
#         loss = loss*(w**BETTA_P)
#         loss = torch.mean(loss)
#         poptimizer.zero_grad()
#         loss.backward()
#         poptimizer.step()

#       else:
#         current_q_values = QValues.get_current(target_net, states, actions)
#         next_q_values = QValues.get_next(policy_net, next_states)
#         target_q_values = (next_q_values*gamma) + rewards

#         loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1), reduction='none')
#         loss = torch.transpose(loss, 0, 1).squeeze()
#         w = torch.from_numpy(memory.get_importance(idx)).to(device)
#         loss = loss*(w**BETTA_P)
#         loss = torch.mean(loss)
#         toptimizer.zero_grad()
#         loss.backward()
#         toptimizer.step()

#       if timestep % priority_update == 0:
#         errors = torch.transpose(current_q_values, 0, 1).squeeze() - target_q_values.detach()
#         memory.refresh_priorities(errors, idx, OFFSET)


#     if em.done:
#       episode_durations.append(timestep)
#       scores.append(score)
#       plot(episode_durations, 100, strategy.rate)
#       current_time = timestep
#       break

#     if best_time < current_time:
#       print("SAVED!")
#       best_time = current_time
#       path = F"/content/gdrive/My Drive/DDQN_1PARAM_SPACE.pt" 
#       torch.save(policy_net.state_dict(), path) 
#       path = F"/content/gdrive/My Drive/DDQN_2PARAM_SPACE.pt"
#       torch.save(target_net.state_dict(), path)

init_episode = 0
for episode in range(num_episodes):
  em.reset()
  state = em.get_state()
  score = 0
  loss_avg = 0

  for timestep in count():
    action = agent.select_action(policy_net, target_net, state)
    reward, info = em.take_action(action)
    score += reward
    next_state = em.get_state()
    memory.push(Experience(state, action, next_state, reward))
    state = next_state
    
    if memory.can_provide_sample(batch_size):
      experiences, idx = memory.sample(batch_size)
      states, actions, next_states, rewards = extract_tensors(experiences)
      current_q_values = QValues.get_current(policy_net, states, actions)
      next_q_values = QValues.get_next(target_net, next_states)
      target_q_values = rewards + next_q_values*gamma

      loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
      poptimizer.zero_grad()
      loss.backward()
      poptimizer.step()

      # loss_avg += loss
      

    if timestep % target_update == 0:
      target_net.load_state_dict(policy_net.state_dict())
      path = "/home/fara-ubuntu/Documents/FARA/dqn/Atari_save/DQN_PARAM_BREAK.pt" 
      torch.save({'policy_net': policy_net.state_dict(),
                  'target_net': target_net.state_dict(),
                  'poptimizer': poptimizer.state_dict(),
                  'episode': episode,
                  'eps_start': strategy.rate,
                  'frames': frames}, path)
      
    if em.done:
      # losses.append(loss_avg/timestep)
      scores.append(score)
      episode_durations.append(timestep)
      frames += timestep
      plot(scores, 100, strategy.rate, init_episode, frames)
      # plot_loss(losses, 100)
      if best_score < score:
        print("SAVED!")
        best_score = score
        path = "/home/fara-ubuntu/Documents/FARA/dqn/Atari_save/DQN_PARAM_BREAK_BEST.pt"  
        torch.save({'policy_net': policy_net.state_dict(),
                   'target_net': target_net.state_dict(),
                   'poptimizer': poptimizer.state_dict(),
                   'episode': episode}, path)
      break

# a = False
# em.reset()
# all_rew = 0
# for i in range(10000):
#   rew, info = em.take_action(torch.tensor([randint(0,3)]))
#   state = em.get_state().to('cpu')
#   # print(info)
#   # plt.figure(1)
#   # plt.clf()
#   # plt.imshow(state[0][0], interpolation = 'none')
#   # plt.title('1')
#   # plt.show()
#   # plt.figure(2)
#   # plt.clf()
#   # plt.imshow(state[0][1].squeeze(0).squeeze(0), interpolation = 'none')
#   # plt.title('2')
#   # plt.show()
#   # plt.figure(3)
#   # plt.clf()
#   # plt.imshow(state[0][2].squeeze(0).squeeze(0), interpolation = 'none')
#   # plt.title('3')
#   # plt.show()
#   # plt.figure(4)
#   # plt.clf()
#   plt.imshow(state[0][3].squeeze(0).squeeze(0), interpolation = 'none')
#   plt.title('4')
#   plt.show()
#   print(state.flatten(start_dim=2).max(dim=2)[0].eq(0))
#   all_rew += rew
#   print(all_rew)
#   if is_ipython: display.clear_output(wait=False)
#   if em.done:
#     print("DONE!")
#     em.reset()
#     print(i)
#     if a is True:
#       # break
#       a = True
#     else: a = True
    


em.close()

