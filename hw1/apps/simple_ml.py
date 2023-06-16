import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname) as f:
        X_fd = f.read()
    
    # reference: https://docs.python.org/3/library/struct.html,
    #            https://xinancsd.github.io/MachineLearning/mnist_parser.html
    # read first four bytes in big endian mode
    _, num_images, rows, cols = struct.unpack_from('>4i', X_fd, 0)
    
    image_pixel =  rows * cols 
    img_fmt = '>' + str(num_images * image_pixel) + 'B'
    
    X = struct.unpack_from(img_fmt, X_fd, offset=16)
    X = np.array(X, dtype=np.float32).reshape(num_images, image_pixel)
    X = X / 255 # normalize

    # read first two bytes
    with gzip.open(label_filename) as f:
        y_fd = f.read()
    _, num_label = struct.unpack_from('>2i', y_fd, 0)
    label_fmt = '>' + str(num_label) + 'B'
    y = struct.unpack_from(label_fmt, y_fd, offset=8)
    y = np.array(y, dtype=np.uint8)
    
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    loss = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,))) # axes better tuple
    
    # here input is one hot tensor
    indexed_logits = ndl.summation(ndl.multiply(Z, y_one_hot), axes=(1,))
    
    loss -= indexed_logits
    loss = ndl.summation(loss, axes=(0,)) / loss.shape[0] # average
    return loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    iters = int(np.ceil(X.shape[0] / batch))
    X_b: ndl.Tensor
    y_b: np.ndarray
    
    for i in range(iters):
        if i == iters - 1: # last batch
            X_b = ndl.Tensor(X[i * batch: , :])
            y_b = y[i * batch:]
        else:
            X_b = ndl.Tensor(X[i * batch: (i + 1) * batch, :])
            y_b = y[i * batch: (i + 1) * batch]
        
        Z = ndl.matmul(ndl.relu(ndl.matmul(X_b, W1)), W2)
        # expand y to 2D tensor
        y_one_hot = np.zeros(Z.shape)
        y_one_hot[np.arange(0, y_b.shape[0], 1), y_b] = 1
        L = softmax_loss(Z, ndl.Tensor(y_one_hot))
    
        L.backward()
        
        W1 -= lr * W1.grad
        W2 -= lr * W2.grad
        # print("iter {} of total {}".format(i, iters))
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
