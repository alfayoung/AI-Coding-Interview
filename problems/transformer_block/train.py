import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from solution import TransformerBlock


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def to(self, device):
        self.data = [{k: v.to(device) for k, v in item.items()} for item in self.data]
        return self

def training_loop(model, data_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        evaluate(model, data_loader)
        for batch in data_loader:
            x = batch['x']
            output = model(x).cpu()
            loss = F.mse_loss(output, batch['y'].cpu())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    
def evaluate(model, data_loader):
    model.eval()
    loss = []
    for batch in data_loader:
        x = batch['x']
        with torch.no_grad():
            output = model(x).cpu()
        loss.append(F.mse_loss(output, batch['y'].cpu()))
    print(f"Eval loss: {np.mean(loss)}")
    

def main():
    dataset = CustomDataset([{'x': torch.randn(10, 8), 'y': torch.randn(10, 8)} for _ in range(20)])
    model = TransformerBlock(8, 2, 4, 8 * 4)
    dataset = dataset.to('cpu')
    model = model.to('cpu')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    training_loop(model, data_loader, optimizer, 5)
    evaluate(model, data_loader)
    
if __name__ == '__main__':
    main()