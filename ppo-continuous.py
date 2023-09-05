import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.95
lmbda           = 0.95
eps_clip        = 0.2
K_epoch         = 100
rollout_len    = 3
buffer_size    = 10
minibatch_size = 32 # size of minibatches of data to use to do a gradient step

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(3,128)
        # note that we share the first layer between all networks but have different layers for v, mu, std network for actions
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        # note that the actor network predicts a gaussian model we can choose to sample continues action from 
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
    
    # make tensor of size (buffer_size * minibatch_size of rollout_len lists)
    def make_batch(self):
        data = []
        # when this is called there are enough (buffer_size * minibatch_size) rollouts in self.data
        # we just move these mini_batches to data and tensorize them
        for j in range(buffer_size):
            s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []

            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
            
            # mini_batch sizes are torch.Size([minibatch_size, rollout_len, shape of s or a or r])
            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                          torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                          torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)
        # data size is list of buffer_size batches of torch.Size([minibatch_size, rollout_len, shape of s or a or r])
        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for batch in data:
            # torch.Size([minibatch_size, rollout_len, shape of s or a or r])
            s, a, r, s_prime, done_mask, old_log_prob = batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            # print(delta.shape, td_target.shape, r.shape)
            # delta.shape = torch.Size([minibatch_size, rollout_len, 1])
            # advantage_lst = []
            # advantage = 0.0
            # for delta_t in delta[::-1]:
            #     advantage = gamma * lmbda * advantage + delta_t[0]
            #     advantage_lst.append([advantage])
            # advantage_lst.reverse()
            # advantage = torch.tensor(advantage_lst, dtype=torch.float)

            advantage = torch.zeros((minibatch_size,3,1), dtype=torch.float)
            for i in reversed(range(rollout_len)):
                # print(delta[:,0].shape)
                if i==rollout_len-1:
                    j=i
                else:
                    j=i+1
                # print(advantage[:,j].shape, delta[:,0].shape)
                advantage[:,i] = gamma * lmbda * advantage[:,j] + delta[:,0].reshape((minibatch_size,1))
                # print(advantage.shape, delta_t.shape, delta.shape, delta[::-1].shape, delta[0], delta[-1], delta[::-1][0], delta[::-1][-1])

            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self):
        # when there are enough data to train do K_Epoch training steps
        if len(self.data) == minibatch_size * buffer_size:
            # print(len(self.data),  minibatch_size , buffer_size)
            # calling this will move current transitions out of self.data to data
            # therefore the train_net() that is called after each rollout wouldn't do anything
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch
                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    # a.shape = torch.Size([minibatch_size, rollout_len, 1])
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob).reshape(-1)# a/b == exp(log(a)-log(b))
                    surr1 = ratio * advantage.reshape(-1)
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage.reshape(-1)
                    # print(ratio.shape, torch.min(surr1, surr2).shape, self.v(s).shape, F.smooth_l1_loss(self.v(s) , td_target).shape)
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s).reshape(-1), td_target.reshape(-1))

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
        # else:
        #     print(len(self.data), len(data), minibatch_size , buffer_size)

def main():
    env = gym.make('Pendulum-v1')
    model = PPO()
    score = 0.0
    print_interval = 20
    rollout = []

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        count = 0
        while count < 200 and not done:
            # again note that we don't wait until the end of episode
            # for each rollout_len transition we make a batch of data and train 
            for t in range(rollout_len):
                mu, std = model.pi(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                s_prime, r, done, truncated, info = env.step([a.item()])

                rollout.append((s, a, r/10.0, s_prime, log_prob.item(), done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                count += 1

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, optmization step: {}".format(n_epi, score/print_interval, model.optimization_step))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()