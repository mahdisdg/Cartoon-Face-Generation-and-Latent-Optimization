import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GANEnvironment:
    def __init__(self, generator, discriminator, classifier, device):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.device = device
        
        # Freeze everything
        for m in [self.generator, self.discriminator, self.classifier]:
            for param in m.parameters():
                param.requires_grad = False
            m.eval()
        
        self.current_z = None
        self.target_meta = None # The features we WANT (e.g. Blonde)
        self.last_total_score = 0
        self.latent_dim = 100

    def reset(self, specific_target=None):
        # 1. Start with random Latent
        self.current_z = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
        
        # 2. Set the Goal (Target Metadata)
        # If no specific target provided, we assume we want to MATCH 
        # the random features of the initial seed (Self-Consistency)
        # OR we could pick a random real metadata vector. 
        if specific_target is not None:
            self.target_meta = specific_target
        else:
            self.target_meta = None 

        # Initial Score calculation
        self.last_total_score = self.get_combined_score(self.current_z)
        
        return self.current_z.squeeze().cpu().numpy()

    def get_combined_score(self, z):
        with torch.no_grad():
            fake_img = self.generator(z)
            
            # 1. Realism Score (0 to 1)
            realism = self.discriminator(fake_img).item()
            
            # 2. Metadata Score
            if self.target_meta is not None:
                # Predict attributes of generated image
                pred_meta_logits = self.classifier(fake_img)
                pred_meta = torch.sigmoid(pred_meta_logits)
                
                # Compare with Target
                match_score = F.cosine_similarity(pred_meta, self.target_meta).item()
            else:
                match_score = 0
                
        # Weighted Sum: 70% Realism, 30% Feature Matching
        return (0.7 * realism) + (0.3 * match_score)

    def step(self, action_idx):
        delta = 0.25
        dim_idx = action_idx % self.latent_dim
        direction = 1 if action_idx < self.latent_dim else -1
        
        # Modify
        self.current_z[0, dim_idx, 0, 0] += (direction * delta)
        self.current_z = torch.clamp(self.current_z, -3.0, 3.0)
        
        # Calculate Reward
        new_total_score = self.get_combined_score(self.current_z)
        
        # Relative Reward
        reward = (new_total_score - self.last_total_score) * 100
        
        self.last_total_score = new_total_score
        next_state = self.current_z.squeeze().cpu().numpy()
        done = False
        
        return next_state, reward, done, {}


class QLearningAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.9, epsilon=0.1):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to('cpu') 
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state_np, force_greedy=False):
        if not force_greedy and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        state_t = torch.FloatTensor(state_np).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.LongTensor([action])
        reward_t = torch.FloatTensor([reward])
        
        q_values = self.q_net(state_t)
        q_value = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.q_net(next_state_t)
            max_next_q = next_q_values.max(1)[0]
            expected_q = reward_t + self.gamma * max_next_q
            
        loss = self.criterion(q_value, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()