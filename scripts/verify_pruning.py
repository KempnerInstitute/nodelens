#!/usr/bin/env python
"""Quick verification script to test that CNN pruning is working correctly."""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

def load_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(test, batch_size=batch_size*2, shuffle=False, num_workers=2))

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def train_model(model, loader, device, epochs=10):
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
    return model

def prune_layer(model, layer_name, layer, indices):
    with torch.no_grad():
        layer.weight.data[indices] = 0
        if layer.bias is not None:
            layer.bias.data[indices] = 0
    # Zero BatchNorm
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if layer_name.replace('conv','bn') in name or layer_name.replace('.conv','.bn') in name:
                with torch.no_grad():
                    m.weight.data[indices] = 0
                    m.bias.data[indices] = 0
                    m.running_mean.data[indices] = 0
                    m.running_var.data[indices] = 1
                break

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, test_loader = load_cifar10()
    
    # Load and train
    print("\nTraining ResNet18 on CIFAR-10...")
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = train_model(model, train_loader, device, epochs=15)
    baseline = evaluate(model, test_loader, device)
    print(f"\nBaseline accuracy: {baseline:.2%}")
    
    # Get conv layers
    convs = [(n,m) for n,m in model.named_modules() if isinstance(m, nn.Conv2d) and m.weight.shape[0]>1]
    print(f"\nTesting pruning on {len(convs)} conv layers...")
    
    # Test: accuracy vs sparsity
    print("\nAccuracy vs Sparsity (random pruning, all layers):")
    for ratio in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
        m = copy.deepcopy(model)
        for name, layer in convs:
            l = dict(m.named_modules())[name]
            n_ch = layer.weight.shape[0]
            n_prune = min(int(n_ch * ratio), n_ch - 1)
            idx = np.random.choice(n_ch, n_prune, replace=False).tolist()
            prune_layer(m, name, l, idx)
        acc = evaluate(m, test_loader, device)
        print(f"  {ratio:.0%}: {acc:.2%} (drop: {baseline-acc:+.2%})")
    
    print("\nIf accuracy drops with higher sparsity, pruning is working!")
    print("If random matches magnitude-based, model is over-parameterized.")

if __name__ == "__main__":
    main()
