"""Identify mnist digits."""
import argparse
import struct
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm


def get_mnist_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Return the mnist test data set in numpy arrays.

    Returns:
        (array, array): A touple containing the test
        images and labels.
    """
    with open("./data/MNIST/t10k-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_test = data.reshape((size, nrows, ncols))

    with open("./data/MNIST/t10k-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_test = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    return img_data_test, lbl_data_test


def get_mnist_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the mnist training data set.

    Returns:
        (array, array): A touple containing the training
        images and labels.
    """
    with open("./data/MNIST/train-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_train = data.reshape((size, nrows, ncols))

    with open("./data/MNIST/train-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    # if gpu:
    #    return cp.array(img_data_train), cp.array(lbl_data_train)
    return img_data_train, lbl_data_train


def normalize(
    data: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """Normalize the input array.

    After normalization the input
    distribution should be approximately standard normal.

    Args:
        data (np.array): The input array.
        mean (float): Data mean, re-computed if None.
            Defaults to None.
        std (float): Data standard deviation,
            re-computed if None. Defaults to None.

    Returns:
        np.array, float, float: Normalized data, mean and std.
    """
    if mean is None:
        # TODO: use np.mean to compute the mean
        mean = 0.0
    if std is None:
        # TODO: use np.std to compute the standard deviation
        std = 0.0

    # TODO: normalize the data: (data - mu)/ sigma
    data_norm = data
    return data_norm, mean, std


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the forward pass."""
        # Set up the neural network
        # See: https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#compact-methods
        # for more information.
        return x


# @jax.jit
def cross_entropy(label: jnp.ndarray, out: jnp.ndarray) -> jnp.ndarray:
    """Compute the cross entropy of one-hot encoded labels and the network output.

    Implement cross_entropy:
    1/n Sum[( -label * log(out) - (1 - label) * log(1 - out) )]

    Args:
        label (jnp.ndarray): The image labels of shape [batch_size, 10].
        out (jnp.ndarray): The network output of shape [batch_size, 10].

    Returns:
        jnp.ndarray, The loss scalar.
    """
    # TODO: Implement me.
    return jnp.array(0.0)


# @jax.jit
def forward_pass(
    variables: FrozenDict, img_batch: jnp.ndarray, label_batch: jnp.ndarray
) -> jnp.ndarray:
    """Do a forward step, by evaluating network and cost function."""
    # TODO: compute the network output. Use cnn.apply .
    out = 0.0
    ce_loss = cross_entropy(nn.one_hot(label_batch, num_classes=10), out)
    return ce_loss


# set up autograd
# TODO: Use jax.value_and_grad
loss_grad_fn = None


# set up SGD
# @jax.jit
def sgd_step(variables, grads, learning_rate):
    """Update the variable in a SGD step."""
    # TODO: use jax.tree_util.tree_map
    return variables


def get_acc(img_data: jnp.ndarray, label_data: jnp.ndarray) -> float:
    """Compute the network accuracy."""
    # TODO: use jnp.argmax
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Networks on MNIST.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    args = parser.parse_args()
    print(args)

    batch_size = 50
    val_size = 1000
    epochs = 10
    img_data_train, lbl_data_train = get_mnist_train_data()
    img_data_val, lbl_data_val = img_data_train[:val_size], lbl_data_train[:val_size]
    img_data_train, lbl_data_train = (
        img_data_train[val_size:],
        lbl_data_train[val_size:],
    )
    # normalize train and validation data.
    img_data_train, mean, std = normalize(img_data_train)
    img_data_val, _, _ = normalize(img_data_val, mean, std)

    exp_list = []

    key = jax.random.PRNGKey(key)  # type: ignore
    cnn = CNN()
    # TODO: initialize the network use
    # cnn.init .

    for e in range(epochs):
        # shuffle the data before every epoch.
        shuffler = jax.random.permutation(key, len(img_data_train))
        img_data_train = img_data_train[shuffler]
        lbl_data_train = lbl_data_train[shuffler]
        img_batches = np.split(
            img_data_train, img_data_train.shape[0] // batch_size, axis=0
        )
        label_batches = np.split(
            lbl_data_train, lbl_data_train.shape[0] // batch_size, axis=0
        )
        
        train_accs = []
        for img_batch, label_batch in tqdm(
            zip(img_batches, label_batches), total=len(img_batches)
        ):
            img_batch, label_batch = (
                jnp.array(np_array) for np_array in (img_batch, label_batch)
            )
            # cel = cross_entropy(nn.one_hot(label_batches[no], num_classes=10),
            #                    out)
            cel, grads = loss_grad_fn(variables, img_batch, label_batch)
            variables = sgd_step(variables, grads, args.lr)
            train_accs.append(get_acc(img_batch, label_batch))
        print("Epoch: {}, loss: {}".format(e + 1, cel))

        train_acc = np.mean(train_accs)
        val_acc = get_acc(img_data_val, lbl_data_val)
        print(
            "Train and Validation accuracy: {:3.3f}, {:3.3f}".format(train_acc, val_acc)
        )

    print("Training done. Testing...")
    img_data_test, lbl_data_test = get_mnist_test_data()
    img_data_test, mean, std = normalize(img_data_test, mean, std)
    test_acc = get_acc(img_data_test, lbl_data_test)
    print("Done. Test acc: {}".format(test_acc))
