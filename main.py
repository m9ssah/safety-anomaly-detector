import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

X = torch.tensor(x, dtype=torch.float32)
i = torch.sensor(y, dtype=torch.float32).unsqueeze(1)

train_size = int(0.8 * N)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
