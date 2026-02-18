# Cartoon Face Generation & Latent Space Optimization with RL

This repository contains a complete, from-scratch implementation of a **Deep Convolutional GAN (DCGAN)** designed for cartoon avatar synthesis, integrated with a **Reinforcement Learning (RL)** agent to navigate and optimize the model's latent space.

Unlike standard generative projects that focus solely on training, this project treats the latent space ($z$) as a searchable environment. We use a Q-Learning agent to "walk" through this space to discover vectors that maximize specific facial attributes and image realism.



## 🚀 Key Features
* **Custom DCGAN Architecture:** Built from the ground up using PyTorch, featuring convolutional transpose layers, batch normalization, and tailored activation functions (ReLU/LeakyReLU).
* **Latent Space Navigation:** Implements a Q-Learning agent that optimizes 100-dimensional latent vectors ($z$) to satisfy external constraints without retraining the generator.
* **Attribute Classification:** A custom CNN-based attribute classifier (trained on 217-dimensional metadata) serves as a reward function for the RL agent.
* **Advanced Evaluation Metrics:** Full implementation of **FID (Frechet Inception Distance)** for image quality and **LPIPS (Learned Perceptual Image Patch Similarity)** for diversity analysis.

## 🛠️ Tech Stack
* **Framework:** PyTorch
* **Computer Vision:** Torchvision, OpenCV, PIL
* **Optimization & RL:** NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn, tqdm

## 📐 Architecture & Methodology

### 1. Generative Adversarial Network (GAN)
The model utilizes a DCGAN architecture to learn the distribution of the CartoonSet100k dataset. 
- **Generator:** Maps a 100D noise vector to a $64 \times 64 \times 3$ image using strided convolutions.
- **Discriminator:** A binary classifier that provides the adversarial loss, improved through one-sided label smoothing to prevent vanishing gradients.

### 2. Reinforcement Learning Environment
After training the GAN, we freeze its weights and treat the Generator as an RL environment:
- **State:** The current latent vector $z$.
- **Action:** Incremental perturbations (steps) in the latent dimensions.
- **Reward:** A composite score based on the Discriminator's realism prediction and the Attribute Classifier's match to a target metadata vector.



## 📊 Evaluation Results
The model was evaluated on 2,000 generated samples:
* **Realism:** The RL agent successfully increased realism scores (e.g., $0.14 \rightarrow 0.29$ on the discriminator scale) by navigating toward higher-density regions of the latent manifold.
* **Diversity:** High LPIPS scores confirm that the model effectively avoids mode collapse, producing a wide range of unique character identities.
* **Accuracy:** The integrated classifier achieved high precision in identifying 217 distinct facial attributes.

## 📁 Project Structure
- `models.py`: Definitions for the Generator and Discriminator.
- `rl_agent.py`: Implementation of the `GANEnvironment` and `QLearningAgent`.
- `classifier.py`: CNN architecture for attribute detection and reward shaping.
- `preprocess.py`: Data pipeline for image normalization and one-hot metadata encoding.
- `evaluate.py`: Logic for FID, LPIPS, and Confusion Matrix generation.
- `train.py`: Orchestration scripts for both GAN adversarial training and RL optimization.

## ⚙️ Setup & Usage
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/your-username/Latent-Flow-GAN-RL.git](https://github.com/mahdisdg/Cartoon-Face-Generation-and-Latent-Optimization.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision numpy pandas matplotlib scikit-learn lpips
    ```
3.  **Run training:**
    Execute the cells in `main.ipynb` or run `train.py` to begin the adversarial training phase followed by RL optimization.
