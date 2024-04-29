# BasicPyNN
Python code for a neural network trained on the MNIST dataset for image recognition. It also features a number prediction game where players draw digits for the network to guess. User to learn basics or neural networks

# Usage
Run the main script (main.py) to train the neural network, play the number prediction game, and test the model's accuracy.
When importing or exporting trained parameters, do not include ".npz" in the name.
Clone the repository to your local machine.
    
Ensure that the MNIST dataset files are present and named as follows:

    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
    train-images-idx3-ubyte
    train-labels-idx1-ubyte

Provided Test Parameters and Training Info

    test1.npz - 10h10e1.00e-2l
    test2.npz - 32h10e1.00e-2l
    test3.npz - 50h20e1.00e-2l

# Note
The tkinter window may open behind other applications.

The training images file are too large to be stored in the repo but can be taken from: http://yann.lecun.com/exdb/mnist/

Dependencies

    NumPy: For mathmatical operations.
    mnist-py: For reading the MNIST dataset.

Install Commands:

    pip install numpy
    pip install mnist-py
