import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import lpips
import subprocess

def evaluate(netG, netD, dataloader, device):
    """
    Performs the final evaluation of the GAN model.
    Includes:
    1. FID Score (Quality)
    2. LPIPS Score (Diversity)
    3. Discriminator Analysis (Accuracy, Confusion Matrix, TP/TN/FP/FN Examples)
    """
    print("\n" + "="*40)
    print("STARTING FINAL EVALUATION")
    print("="*40)

    # Ensure models are in eval mode
    netG.eval()
    netD.eval()

    # Create temporary directories for metrics
    real_path = "temp_eval_real"
    fake_path = "temp_eval_fake"
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)

    # ==========================================
    # PART 1: PREPARE IMAGES (Required for FID)
    # ==========================================
    print("\n[1/4] Preparing Images for Evaluation...")
    
    # Save 500 Real Images
    subset_indices = range(500)
    temp_loader = DataLoader(Subset(dataloader.dataset, subset_indices), batch_size=1)
    
    for i, data in enumerate(temp_loader):
        vutils.save_image(data[0], os.path.join(real_path, f"real_{i}.png"), normalize=True)
        
    # Save 500 Fake Images
    with torch.no_grad():
        for i in range(500):
            z = torch.randn(1, 100, 1, 1, device=device)
            fake = netG(z)
            vutils.save_image(fake, os.path.join(fake_path, f"fake_{i}.png"), normalize=True)

    # ==========================================
    # PART 2: FID SCORE (Visual Quality)
    # ==========================================
    print("\n[2/4] Calculating FID Score (Fréchet Inception Distance)...")
    try:
        # We call the library via subprocess to avoid complex import dependencies
        result = subprocess.run(
            ["python", "-m", "pytorch_fid", real_path, fake_path, "--device", str(device).split(":")[0]],
            capture_output=True, text=True
        )
        fid_output = result.stdout.strip()
        # Parse the number from the output string "FID:  123.45"
        print(f"   >> {fid_output}")
    except Exception as e:
        print(f"   >> Error calculating FID: {e}")
        print("   >> Ensure 'pytorch-fid' is installed (pip install pytorch-fid).")

    # ==========================================
    # PART 3: LPIPS SCORE (Diversity)
    # ==========================================
    print("\n[3/4] Calculating LPIPS Score (Diversity)...")
    try:
        loss_fn = lpips.LPIPS(net='alex').to(device)
        scores = []
        pairs = []

        with torch.no_grad():
            for i in range(50): # Check 50 random pairs
                z1 = torch.randn(1, 100, 1, 1, device=device)
                z2 = torch.randn(1, 100, 1, 1, device=device)
                img1, img2 = netG(z1), netG(z2)
                
                # Compute distance
                dist = loss_fn(img1, img2).item()
                scores.append(dist)
                
                # Store for visualization (First 4 pairs)
                if i < 4:
                    pairs.append((img1, img2, dist))
        
        avg_lpips = sum(scores) / len(scores)
        print(f"   >> Average LPIPS: {avg_lpips:.4f} (Higher is better, <0.05 is collapse)")
        
        # Visualize Diversity Examples
        plt.figure(figsize=(8, 2))
        for idx, (im1, im2, d) in enumerate(pairs[:4]):
            plt.subplot(1, 4, idx+1)
            # Stitch images side-by-side
            combined = torch.cat((im1, im2), dim=3)
            plt.imshow(combined[0].permute(1,2,0).cpu() * 0.5 + 0.5)
            plt.axis('off')
            plt.title(f"Dist: {d:.2f}", fontsize=8)
        plt.suptitle("LPIPS Diversity Samples (Pairs)")
        plt.show()
        
    except Exception as e:
        print(f"   >> Error calculating LPIPS: {e}")

    # ==========================================
    # PART 4: DISCRIMINATOR ANALYSIS
    # ==========================================
    print("\n[4/4] Analyzing Discriminator Performance...")
    
    y_true, y_pred = [], []
    examples = {"TP": [], "TN": [], "FP": [], "FN": []}
    
    with torch.no_grad():
        # Check 10 batches
        for i, data in enumerate(dataloader):
            if i >= 10: break
            
            # Real
            real = data[0].to(device)
            out_r = netD(real)
            pred_r = (out_r > 0.5).float()
            
            # Fake
            z = torch.randn(real.size(0), 100, 1, 1, device=device)
            fake = netG(z)
            out_f = netD(fake)
            pred_f = (out_f > 0.5).float()
            
            # Collect Stats
            y_true.extend([1]*len(real) + [0]*len(fake))
            y_pred.extend(pred_r.cpu().tolist() + pred_f.cpu().tolist())
            
            # Collect Examples (Limit 1 per category to save memory)
            for j in range(len(real)):
                if len(examples["FN"]) < 3 and pred_r[j] == 0: examples["FN"].append(real[j])
                if len(examples["TP"]) < 3 and pred_r[j] == 1: examples["TP"].append(real[j])
            
            for j in range(len(fake)):
                if len(examples["FP"]) < 3 and pred_f[j] == 1: examples["FP"].append(fake[j])
                if len(examples["TN"]) < 3 and pred_f[j] == 0: examples["TN"].append(fake[j])

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"   >> Discriminator Accuracy: {acc*100:.2f}%")
    
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Fake', 'Pred Real'], yticklabels=['True Fake', 'True Real'])
    plt.title("Confusion Matrix")
    plt.show()
    
    # Plot Classification Examples
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    category_map = [("TP (Real->Real)", examples["TP"]), ("TN (Fake->Fake)", examples["TN"]),
                    ("FP (Fake->Real)", examples["FP"]), ("FN (Real->Fake)", examples["FN"])]
    
    for ax, (title, imgs) in zip(axes.flatten(), category_map):
        if len(imgs) > 0:
            ax.imshow(imgs[0].permute(1,2,0).cpu() * 0.5 + 0.5)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.suptitle("Discriminator Classification Examples")
    plt.tight_layout()
    plt.show()

    # Cleanup
    shutil.rmtree(real_path)
    shutil.rmtree(fake_path)
    print("Evaluation Complete. Temporary files removed.")