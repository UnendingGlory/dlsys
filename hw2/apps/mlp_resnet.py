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
                             nn.Linear(hidden_dim, dim), norm(dim)
                             )
    residual = nn.Residual(backbone)
    return nn.Sequential(residual, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # * means decode
    res = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
                        nn.Linear(hidden_dim, num_classes)
                        )
    return res
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # opt is optimizer
    losses, err = [], 0
    loss_fn = nn.SoftmaxLoss()
    if opt:
        model.train()
        for X, y in dataloader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            losses.append(loss.numpy())
            err += np.sum(y_hat.numpy().argmax(axis=1) != y.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    else:
        model.eval()
        for X, y in dataloader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            losses.append(loss.numpy())
            err += np.sum(y_hat.numpy().argmax(axis=1) != y.numpy())
    return err / len(dataloader.dataset), np.mean(losses)
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(28 * 28, hidden_dim=hidden_dim)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                                          os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    train_dataloader = ndl.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    for i in range(epochs):
        train_err, train_loss = epoch(train_dataloader, resnet, opt=opt)
        # print("epoch {} finish, train_err: {}, train_loss: {}".format(i, train_err, train_loss))
        
    test_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                                          os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    test_err, test_loss = epoch(test_dataloader, resnet)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
