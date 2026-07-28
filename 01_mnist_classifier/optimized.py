"""
Optimized MNIST classifier — the baseline, whittled with int8 quantization.
"""

import torch.quantization
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from baseline import Net



def load_baseline():
    model = Net()
    model.load_state_dict(torch.load("mnist_baseline.pt", map_location= "cpu"))
    model.eval()
    return model

def quantize(model):
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

def main():
    tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])

    model = load_baseline()
    qmodel = quantize(model)
    torch.save(qmodel.state_dict(), "mnist_int8.pt")
    print("saved int8 weights -> mnist_int8.pt")
    test_dataset  = datasets.MNIST(root='data', train=False, download=True, transform = tfm)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)


    

    correct = 0
    total = 0


    for images, labels in test_loader:
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = qmodel(images)
        preds = outputs.argmax(dim = 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"test accuracy {correct / total:.4f}")



if __name__ == "__main__":
    main()

