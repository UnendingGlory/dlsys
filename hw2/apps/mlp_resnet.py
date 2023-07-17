import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    backbone = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), 
                             nn.ReLU(), nn.Dropout(drop_prob), 
                             nn.Liear(hidden_dim, dim), norm(dim)
                             )
    residual = nn.Residual(backbone)
    return nn.Sequential(residual, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # * means decode
    res = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
                        nn.Linear(dim, num_classes)
                        )
    return res
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # opt is optimizer
    loss, err = 0, 0
    loss_fn = nn.SoftmaxLoss()
    if opt:
        model.train()
        for X, y in dataloader:
            y_hat = model(X)
            loss += loss_fn(y_hat, y)
            err += np.sum(y_hat.numpy().argmax(axis=1) != y)
            opt.reset_grad()
            loss.backward()
            opt.step()
    else:
        model.eval()
        for X, y in dataloader:
            y_hat = model(X)
            loss += loss_fn(y_hat, y)
            err += np.sum(y_hat.numpy().argmax(axis=1) != y)
    return err / len(dataloader.dataset), loss / len(dataloader)
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(28 * 28, hidden_dim=hidden_dim)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    for i in range(epochs):
        epoch()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
