# TensorFlow MNIST

This repository is going to use TensorFlow to train the MNIST network.

---
## File Structure

```
mnist/
    |--- out/                       # For the ouput files
    |--- src/                       # Source code are in this directory
        |--- mnist_fully.py         # 2-Hidden Layer Fully Connected NN with TensorFlow
        |--- mnist_fully_model.py   # Add save and restore in mnist_fully.py
    |--- README.md      
    |--- LICENSE
    |--- .gitignore     
```

---
## Installation

> The following instructions are for installing on **Ubuntu Linux 16.04**

### TensorFlow

> TensorFlow is tested and supported on the following 64-bit systems:
> * Ubuntu 16.04 or later
> * Windows 7 or later
> * macOS 10.12.6 (Sierra) or later (no GPU support)
> * Raspbian 9.0 or later

1. Install TensorFlow
    > For Python 2.7, you can follow the instructions [here](https://www.tensorflow.org/install/pip?lang=python2).
    * Prerequisite (for **Python 3 (Python 3.4, 3.5, 3.6)**)
        ```bash
        # Check if your Python environment is already configured
        $ python3 --version
        $ pip3 --version
        $ virtualenv --version
        # If the above packages are already installed, skip to the next step
        $ sudo apt-get update
        $ sudo apt-get install python3-dev python3-pip
        # Install for system-wide
        $ sudo pip3 install -U virtualenv
        ```
    * Create a virtual environment (recommended)
        ```bash
        # Create a new vitrual environment by choosing a Python interpreter and making a ./env directory to hold it
        $ virtualenv --system-site-packages -p python3 ./venv
        # Activate the virtual environement using a shell-specific command (e.g., sg, bash, etc.)
        $ source ./venv/bin/activate
        # When virtualenv is active, your shell prompt is prefixed with (venv).
        (venv) $
        # Install packages within a virtual environment without affecting the host system setup. Start by upgrading pip:
        (venv) $ pip install --upgrade pip
        # Show packages installed within the virtual environment
        (venv) $ pip list
        # To exit virtualenv later
        (venv) $ deactivate
        ```
    * Install TensorFlow with Python's pip package manager 
        ```bash
        # Current release for GPU-only (Python 2.7)
        (venv) $ pip install --upgrade tensorflow
        # GPU package for CUDA-enabled GPU cards (Python 2.7)
        (venv) $ pip install --upgrade tensorflow-gpu
        ```
2. Run the example program `hello.py`
    ```bash
    # Make sure your current directory is src/
    (venv) $ python hello.py
    b'Hello, TensorFlow!'
    ```

---
## Execution

* `./src/mnist_fully.py`
    ```bash
    $ python mnist_fully.py
    2018-12-11 18:57:18.102112: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    Step 1, Minibatch Loss= 8470.1895, Training Accuracy= 0.328
    Step 100, Minibatch Loss= 250.3591, Training Accuracy= 0.875
    Step 200, Minibatch Loss= 173.7493, Training Accuracy= 0.828
    Step 300, Minibatch Loss= 155.4185, Training Accuracy= 0.820
    Step 400, Minibatch Loss= 81.8946, Training Accuracy= 0.859
    Step 500, Minibatch Loss= 18.5791, Training Accuracy= 0.938
    Optimization Finished!
    Testing Accuracy: 0.8563
    ```
* `./src/mnist_fully_model.py`
    ```bash
    $ python mnist_fully_model.py
    2018-12-11 19:18:36.799674: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    Step 1, Minibatch Loss= 9184.1309, Training Accuracy= 0.391
    Step 100, Minibatch Loss= 365.5018, Training Accuracy= 0.852
    Step 200, Minibatch Loss= 153.3021, Training Accuracy= 0.820
    Step 300, Minibatch Loss= 24.0616, Training Accuracy= 0.898
    Step 400, Minibatch Loss= 56.0503, Training Accuracy= 0.875
    Step 500, Minibatch Loss= 74.1631, Training Accuracy= 0.844
    Optimization Finished!
    Model saved in file: /tmp/model.ckpt
    Model restored from file: /tmp/model.ckpt
    Testing Accuracy: 0.8559
    ```
* `./src/mnist_fully_input.py`
    ```bash
    $ python mnist_fully_input.py
    2018-12-11 19:25:24.784290: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    Model restored from file: /tmp/model.ckpt
    Answer: [7 2 1 ..., 4 5 6]
    ```

---
## Contributor

* [David Lu](https://gitbib.com/yungshenglu)

---
## License

Apache License 2.0