import torch
import torch.nn as nn
import torch.optim as optim

class SimpleAttributeClassifier(nn.Module):
    def __init__(self, num_attributes=217):
        super(SimpleAttributeClassifier, self).__init__()
        
        # Simple CNN to extract features
        self.features = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 32, 4, 2, 1), # -> 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 4, 2, 1), # -> 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 4, 2, 1), # -> 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)) # -> 128 x 1 x 1
        )
        
        # Classifier Head
        self.classifier = nn.Linear(128, num_attributes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def train_classifier(dataloader, model, device, epochs=3):
    """
    Fast training for the attribute classifier.
    """
    criterion = nn.MSELoss() # Using MSE because our One-Hot vectors are like probabilities
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    print(f"Training Attribute Classifier for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for images, metadata in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Use Sigmoid to force outputs to 0-1 range for MSE comparison
            loss = criterion(torch.sigmoid(outputs), metadata)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Classifier Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
    
    model.eval()
    return model