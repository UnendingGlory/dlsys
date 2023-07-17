import struct
import gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        return np.fliplr(img) if flip_img else img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        height, width, _ = img.shape
        return padded_img[shift_x + self.padding: height + shift_x + self.padding, 
                          shift_y + self.padding: width + shift_y + self.padding,
                          :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size)) # normal ordering of all batches

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.idx = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.idx >= len(self.ordering):
            raise StopIteration
        batch_idx = self.ordering[self.idx]
        samples = self.dataset[batch_idx]
        self.idx += 1
        return [Tensor(sample) for sample in samples]
    
        ### END YOUR SOLUTION


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


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms=transforms)
        self.height, self.width, self.channel = 28, 28, 1 # mnist is grayscale, so channel is 1
        self.img, self.label = parse_mnist(image_filename, label_filename)
        self.data: List
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # index may be an integer or a list
        if len(self.img[index].shape) > 1:
            tform_images = [self.apply_transforms(img.reshape(self.height, self.width, self.channel)) \
                        for img in self.img[index]]
        
            tform_images = np.vstack(tform_images)
        else:
            index_img = self.img[index].reshape(self.height, self.width, self.channel)
            tform_images = self.apply_transforms(index_img)    
        
        tform_images = tform_images.reshape((-1, self.height * self.width))
        return (tform_images, self.label[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.img)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
