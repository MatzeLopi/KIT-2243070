Of course\! After analyzing the files in your `models` folder, I've updated the README to include a detailed description of your unified API.

Here is the new and improved `README.md`:

-----

# Data-Based Modeling and Control

## 👋 Hello there\!

Welcome to my repository for the Data-Based Modeling and Control course at my university. This project contains my solutions to the course assignments, where I've explored various concepts and techniques in this exciting field.

The primary goal of this repository is to serve as a personal archive of my work and to showcase the application of modern machine learning libraries like JAX and Flax to control theory problems.

## 📚 Topics Covered

This course covered a range of topics in data-based modeling and control. The solutions in this repository touch upon:

  * **ARX Models, Linear and Nonlinear Systems (Exercises 1 & 2)**: Exploration of AutoRegressive with eXogenous input models for both linear and nonlinear systems.
  * **Hammerstein Models (Exercise 3)**: Implementation of models consisting of a static nonlinear block followed by a linear dynamic system.
  * **Dynamic Mode Decomposition (DMD) (Exercise 4)**: Application of DMD for data-driven modeling and analysis of dynamical systems.
  * **Dynamic Mode Decomposition with Control (DMDC) (Exercise 5)**: Extending DMD to incorporate the effects of control inputs.
  * **Neural Networks (Exercise 6)**: Utilizing neural networks for system identification and control.

Feel free to browse the different folders to see the implementations for each assignment.

## 🤖 Unified API

In the `models` folder, you'll find a unified API that I created to streamline the process of working with the different models in this project. While there isn't a strict base class, each model follows a consistent design pattern, providing a clean and predictable interface.

Each model module (`arx.py`, `dmd.py`, etc.) exposes three main functions:

  * **`fit(...)`**: This function takes the training data (and model-specific parameters) and returns the learned model parameters. For example, in the ARX model, it returns the parameter vector and the covariance matrix. For DMD, it returns the reduced-order model, dynamics, eigenvalues, and transformation matrix.
  * **`simulate(...)`**: Given the initial conditions and an input sequence, this function uses the learned model parameters to predict the system's future behavior.
  * **`optimize(...)`**: This is a higher-level function that searches for the optimal model hyperparameters (e.g., `na` and `nb` for ARX, or the rank `r` for DMD/DMDC). It systematically tests a range of values, uses the `fit` and `simulate` functions to evaluate each combination, and returns the best performing model based on the mean absolute error.

### Example Usage (ARX Model)

Here’s a quick example of how you might use the API for an ARX model:

```python
import jax.numpy as jnp
from models import arx, utils

# 1. Load your data
data = utils.read_mat('your_data.mat')
y_train, u_train = data['y_train'], data['u_train']
y_val, u_val = data['y_val'], data['u_val']

# 2. Find the best model orders
best_params, loss, best_na, best_nb = arx.optimize(
    y=y_train,
    u=u_train,
    na_range=(1, 10),
    nb_range=(1, 10)
)

print(f"Optimal orders: na={best_na}, nb={best_nb} with loss={loss}")

# 3. Simulate the model on new data
y_initial = y_val[:best_na]
y_predicted = arx.simulate(
    y0=y_initial,
    u=u_val,
    na=best_na,
    nb=best_nb,
    param_vector=best_params
)
```

This consistent structure makes it easy to switch between different models and compare their performance on the same dataset.

## 💻 Technologies Used

The solutions in this repository are primarily written in Python and leverage the following libraries:

  * **[JAX](https://github.com/google/jax)**: For high-performance numerical computing and machine learning research.
  * **[Flax](https://github.com/google/flax)**: A neural network library and ecosystem for JAX that is designed for flexibility.
  * **[SciPy](https://scipy.org/)**: For scientific and technical computing.
  * **[Matplotlib](https://matplotlib.org/)**: For creating static, animated, and interactive visualizations.

## Happy Coding\! 🎉