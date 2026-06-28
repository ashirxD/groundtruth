import torch
import torch.nn as nn

# 1. Simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)


# 2. Create model
model = SimpleNet()

# 3. Fake input (batch of 2 samples, 3 features each)
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# 4. Forward pass
output = model(x)

print("Input shape:", x.shape)
print("Output:", output)