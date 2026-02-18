import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
from rl_agent import GANEnvironment, QLearningAgent

def train_gan(dataloader, netG, netD, device, epochs=5, lr=0.0002, beta1=0.5):
    """
    Trains the GAN using a standard adversarial loop with one-sided label smoothing.
    
    Args:
        dataloader: PyTorch DataLoader containing training data.
        netG: Generator model.
        netD: Discriminator model.
        device: Calculation device (CPU/GPU).
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizers.
        beta1: Beta1 hyperparameter for Adam optimizers.
    
    Returns:
        tuple: Lists of Generator losses and Discriminator losses.
    """
    
    criterion = nn.BCELoss()
    
    # Initialize optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise for visualizing generation progress consistency
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    # Label smoothing configuration
    # Using 0.9 for real labels helps prevent the discriminator from becoming overconfident
    real_label_val = 0.9
    fake_label_val = 0.0
    
    # Ensure directories exist
    os.makedirs("model_artifacts", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    print(f"Starting GAN training for {epochs} epochs...")
    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, data in pbar:
            real_images = data[0].to(device)
            b_size = real_images.size(0)

            # ============================================
            # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # ============================================
            netD.zero_grad()
            
            # Train with Real Batch
            label = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with Fake Batch
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake_images = netG(noise)
            
            label.fill_(fake_label_val)
            output = netD(fake_images.detach()) 
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Update D
            errD = errD_real + errD_fake
            optimizerD.step()

            # ============================================
            # 2. Update Generator: maximize log(D(G(z)))
            # ============================================
            netG.zero_grad()
            
            # Generator objective: fool D into outputting 1 (Real)
            label.fill_(1.0) 
            
            output = netD(fake_images)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            # Update G
            optimizerG.step()

            # Logging
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            pbar.set_postfix({
                "Loss_D": f"{errD.item():.4f}", 
                "Loss_G": f"{errG.item():.4f}",
                "D(x)": f"{D_x:.2f}",
                "D(Gz)": f"{D_G_z1:.2f}"
            })

        # End of epoch handling
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        
        vutils.save_image(fake, f"samples/epoch_{epoch}_samples.png", padding=2, normalize=True)
        torch.save(netG.state_dict(), f"model_artifacts/generator_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"model_artifacts/discriminator_epoch_{epoch}.pth")

    return G_losses, D_losses

def train_rl_with_metadata(netG, netD, classifier, dataloader, device, episodes=500):
    """
    Trains the Reinforcement Learning agent to optimize latent vectors
    based on both realism (Discriminator) and metadata targets (Classifier).
    """
    env = GANEnvironment(netG, netD, classifier, device)
    
    # Action space: 100 dimensions * 2 directions (increase/decrease)
    agent = QLearningAgent(state_dim=100, action_dim=200)
    
    rewards_history = []
    
    # Pre-fetch a set of real metadata vectors to use as targets during training
    # This ensures the agent learns to target realistic attribute combinations
    real_targets = []
    for _, meta in dataloader:
        real_targets.append(meta)
        if len(real_targets) > 50: 
            break
    
    pbar = tqdm(range(episodes), desc="Training RL Agent")
    
    for episode in pbar:
        # Select a random target from the real distribution
        target = real_targets[episode % len(real_targets)].to(device)[0].unsqueeze(0)
        
        state = env.reset(specific_target=target)
        total_reward = 0
        
        # Optimization loop (limit to 20 steps per episode)
        for step in range(20):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            loss = agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            avg_rew = np.mean(rewards_history[-10:])
            pbar.set_postfix({"Avg Reward": f"{avg_rew:.2f}"})
            
    return agent, rewards_history