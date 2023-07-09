# code interpreter output for prompt:
#
# share code to train a PyTorch model on this dataset where 
# the trained model would be able to predict whether a 
# given listing would have a price above or below the average 
# price. Include a way to exercise the trained model to predict 
# whether new data points have a price that is above or below the average.



import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# Assuming that `X_train` and `y_train` are your training data and labels




# Define a PyTorch Dataset
class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
        
    def __len__(self):
        return len(self.data)

# Create DataLoaders
train_dataset = MyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Define a loss function and an optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(100):
    for i, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        outputs = model(data)
        loss = criterion(outputs, targets.view(-1, 1))
        
        loss.backward()
        optimizer.step()

# To make predictions
def predict(model, data):
    data = torch.tensor(data, dtype=torch.float32)
    outputs = model(data)
    return (outputs > 0.5).float()
