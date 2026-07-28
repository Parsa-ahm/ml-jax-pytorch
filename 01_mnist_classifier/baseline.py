"""
Baseline MNIST classifier — the standard PyTorch CNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

#Model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():

    #Data
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform = tfm)
    test_dataset  = datasets.MNIST(root='data', train=False, download=True, transform = tfm)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

    #Train

    EPOCHS = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(EPOCHS):
        correct, total, running_loss = 0, 0, 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}")
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # live stats — running train accuracy + average loss, updated in place
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            bar.set_postfix(acc=correct / total, loss=running_loss / total)

    #Eval
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim = 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"test accuracy {correct / total:.4f}")


    #Save

    torch.save(model.state_dict(), "mnist_baseline.pt")
    print("saved weights -> mnist_baseline.pt")

if __name__ == "__main__":
    main()
