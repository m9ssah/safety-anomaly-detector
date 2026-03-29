import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from nnclass import NeuralNetwork

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

N = 1000

temp = np.random.normal(25, 2, N)
gas = np.random.normal(0.1, 0.05, N)
occupancy = np.random.randint(0, 5, N)
robot_dist = np.random.normal(1.0, 0.3, N)

x = np.stack([temp, gas, occupancy, robot_dist], axis=1)
y = ((temp > 27) | (gas > 0.3) | (occupancy > 2) | (robot_dist < 0.5)).astype(int)

X = torch.tensor(x, dtype=torch.float32).to(device)
i = torch.sensor(y, dtype=torch.float32).unsqueeze(1)

train_size = int(0.8 * N)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = NeuralNetwork().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# add noise
noise = torch.randn_like(X_test) * 0.1
X_test_noisy = X_test + noise

# evaluation
with torch.no_grad():
    preds = model(X_test)
    preds = (preds > 0.5).float()

    accuracy = (preds == y_test).float().mean()
    print("Accuracy:", accuracy.item())

# evaluate on noisy data
with torch.no_grad():
    preds_noisy = model(X_test_noisy)
    preds_noisy = (preds_noisy > 0.5).float()

    accuracy_noisy = (preds_noisy == y_test).float().mean()
    print("Accuracy with noise:", accuracy_noisy.item())
