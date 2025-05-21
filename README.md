# Neural Network from Scratch in NumPy

This repository contains an implementation of a basic feedforward neural network using only NumPy, created as part of a coursework assignment. It demonstrates key concepts of neural networks, including forward propagation, backpropagation, and training using stochastic gradient descent.

## Files

- **`nnet.py`**  
  Implements the core logic for a multi-layer perceptron (MLP), including:
  - Parameter initialization
  - Forward and backward propagation
  - Loss computation (cross-entropy)
  - Weight updates using gradients
  - Network evaluation on test data

- **`Main.ipynb`**  
  A Jupyter notebook that:
  - Loads and processes the MNIST digit classification dataset
  - Initializes the neural network using `nnet.py`
  - Trains the model for a fixed number of epochs
  - Tracks training and test accuracy
  - Plots learning curves and analyzes model performance

## Features

- Neural network implemented **from scratch using NumPy**
- Fully-connected architecture with configurable layer sizes
- ReLU activation for hidden layers, softmax for output
- Trains on a subset of **MNIST** with minimal preprocessing
- Accuracy visualization and performance analysis included

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/neural-net-numpy.git
   cd neural-net-numpy
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Main.ipynb
   ```

4. Run all cells to train and evaluate the model.

## Sample Results

The model achieves approximately **92–94% accuracy** on the test set after training for 5 epochs with a small subset of MNIST (first 10,000 examples).

Training and test accuracy over epochs:

Look at plot.png and plot1.png

## Notes

- This is a **learning-focused** implementation and does not include advanced optimization techniques like momentum, dropout, or learning rate schedules.
- The code is written for educational clarity over performance.

## License

This project is intended for academic and educational use.

---

Feel free to use or adapt this project for learning or teaching purposes. Contributions and improvements are welcome!

## Sample Output Summary

### ✅ XOR Network Testing
```
Input:
[[0 1]
 [1 0]
 [1 1]]

Ground Truth:
[[1]
 [1]
 [0]]

Predictions:
[[1]
 [1]
 [0]]
```

The XOR neural network correctly predicted the binary outcomes, demonstrating successful training on a non-linearly separable problem.

### 🤖 Stress Classification Test

- Test set contained biometric-like features (e.g., [heart rate, skin conductance]).
- Network produced accurate predictions on 150+ instances.
- Ground truth and predictions matched for nearly all entries, showing the model generalizes well.

### 🔁 Training Snapshot
```
Forward Propagation Output (y): [[0.83056853]]
Epoch 0, Train Loss: 0.0287
Validation Loss: 1.80e-08
```

### ✏️ Manual Parameter Updates (Verification)

Weights `H2` and bias `B2` were manually updated and verified post-backpropagation:
```
H2: [[1.85946238], [1.15647869], [0.85946238], [1.65647869]]
B2: [[1.05100347]]
```

These match the network-updated values, validating correctness of the backprop implementation.
