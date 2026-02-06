import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_autoencoder(model, X_train, epochs=40, batch_size=2048, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataset = TensorDataset(torch.FloatTensor(X_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
    return model

def get_reconstruction_error(model, X_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs = torch.FloatTensor(X_data).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        mse = torch.mean((inputs - outputs) ** 2, dim=1).cpu().numpy()
    return mse