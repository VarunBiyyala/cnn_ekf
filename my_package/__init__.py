
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import jax
import random
import seaborn as sns
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import tensorflow as tf


# Helper function that visualizes 2d posterior predictive distribution
def plot_posterior_predictive(ax, X, Y, title, Xspace=None, Zspace=None, cmap=cm.rainbow):
    if Xspace is not None and Zspace is not None:
        ax.contourf(*(Xspace.T), (Zspace.T[0]), cmap=cmap, levels=50)
        ax.axis('off')
    colors = ['red' if y else 'blue' for y in Y]
    ax.scatter(*X.T, c=colors, edgecolors='black', s=50)
    ax.set_title(title)
    return ax

# Generate spiral dataset
# Adapted from https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5
def generate_spiral_dataset(key=0, num_per_class=250, zero_var=1., one_var=1., shuffle=True):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key1, key2, key3, key4 = jr.split(key, 4)

    theta = jnp.sqrt(jr.uniform(key1, shape=(num_per_class,))) * 2*jnp.pi
    r = 2*theta + jnp.pi
    generate_data = lambda theta, r: jnp.array([jnp.cos(theta)*r, jnp.sin(theta)*r]).T

    # Data for output zero
    zero_input = generate_data(theta, r) + zero_var * jr.normal(key2, shape=(num_per_class, 2))
    zero_output = jnp.zeros((num_per_class, 1,))

    # Data for output one
    one_input = generate_data(theta, -r) + one_var * jr.normal(key3, shape=(num_per_class, 2))
    one_output = jnp.ones((num_per_class, 1,))

    # Stack the inputs and standardize
    input = jnp.concatenate([zero_input, one_input])
    input = (input - input.mean(axis=0)) / input.std(axis=0)

    # Generate binary output
    output = jnp.concatenate([jnp.zeros(num_per_class), jnp.ones(num_per_class)])

    if shuffle:
        idx = jr.permutation(key4, jnp.arange(num_per_class * 2))
        input, output = input[idx], output[idx]

    return input, output

# Function to that returns the posterior predictive probability for each point in grid
def posterior_predictive_grid(grid, mean, apply, binary=False):
    inferred_fn = lambda x: apply(mean, x)
    fn_vec = jnp.vectorize(inferred_fn, signature='(2)->(3)')
    Z = fn_vec(grid)
    if binary:
        Z = jnp.rint(Z)
    return Z

# Function that plots the convergence of filtered estimates to the batch MAP estimate
def plot_weight_convergence(mean_hist, cov_hist, legend_font_size=14):
    input_dim = mean_hist[-1].shape[0]
    tau_hist = jnp.array([cov_hist[:, i, i] for i in range(input_dim)]).T
    elements = (mean_hist.T, tau_hist.T)
    n_datapoints = len(mean_hist)
    timesteps = jnp.arange(n_datapoints) + 1

    num_weights = 10
    num_cols = 1

    fig, axs = plt.subplots(num_weights, num_cols, figsize=(20, num_weights * 4))

    for k, (wk, Pk) in enumerate(zip(*elements)):
        if k >= num_weights:
            break
        ax = axs[k]
        ax.errorbar(timesteps, wk, jnp.sqrt(Pk), c="blue", label = f"$w_{{{k}}}$")

        ax.set_xlim(1, n_datapoints)

        ax.set_xlabel("ordered sample number", fontsize=15)
        ax.set_ylabel("weight value", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=15)
        sns.despine()
        ax.legend(frameon=False, fontsize=legend_font_size)

    # Remove any unused subplots
    if num_weights < len(axs):
        for j in range(num_weights, len(axs)):
            fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

# Function that prints all of the weights per layer in the model
def plot_weights(model_dims, mean_hist, cov_hist, num_per_layer):
    mean_T = mean_hist.T
    tau_T = jnp.array([cov_hist[:, i, i] for i in range(mean_hist.shape[-1])]).T.T
    n_datapoints = len(mean_hist)
    timesteps = jnp.arange(n_datapoints) + 1

    num_rows = len(model_dims) - 1
    num_cols = 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))

    weights = []
    for i in range(len(model_dims) - 1):
        weights.append(model_dims[i] * model_dims[i+1] + model_dims[i+1])

    sum_weights = [weights[0]]
    for i in range(1, len(weights)):
        new_element = sum_weights[i-1] + weights[i]
        sum_weights.append(new_element)
    sum_weights.insert(0,0)

    for i in range(num_rows):
        ax = axs[i]
        for j in range (num_per_layer):
            idx = random.randint(sum_weights[i], sum_weights[i+1] - 1)
            color = (np.random.random(), np.random.random(), np.random.random())
            ax.set_xlim(1, n_datapoints)
            ax.errorbar(timesteps, mean_T[idx], tau_T[idx], c=color, label = f"$w_{{{idx}}}$")

        ax.set_xlabel("ordered sample number")
        ax.set_ylabel("Weight Value")
        ax.set_title(f"Layer {i + 1}")
        sns.despine()
        ax.legend()

    plt.tight_layout()
    plt.show()

#Generate rotated data for extended animation
def generate_rotated_dataset(rot = 0, key=0, num_per_class=250, zero_var=1., one_var=1., shuffle=True):
  input, output = generate_spiral_dataset(key, num_per_class, zero_var, one_var, shuffle)

  theta = 2.0*jnp.pi *rot/360.0
  rot_mat = jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])
  input_rot = input @ rot_mat
  # input_rot_with_bias = jnp.concatenate([jnp.ones((num_per_class, 1)), input_rot], axis=1)

  return input_rot, output

# Function that returns MNIST data filtered on classes passed as input
def load_mnist_data_filtered(classes_to_include, convert_binary = False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Create masks based on the selected classes
    train_mask = np.isin(y_train, classes_to_include)
    test_mask = np.isin(y_test, classes_to_include)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    if convert_binary:
       y_train = np.where(y_train == sorted(classes_to_include)[0], 0, 1)

    # Convert to JAX arrays
    X_train, y_train = jnp.array(X_train), jnp.array(y_train).astype(jnp.int32)
    X_test, y_test = jnp.array(X_test), jnp.array(y_test).astype(jnp.int32)

    # Verify the results
    print(f"Training set shape: {X_train.shape}, Labels: {jnp.unique(y_train)}")
    print(f"Test set shape: {X_test.shape}, Labels: {jnp.unique(y_test)}")

    return X_train, y_train, X_test, y_test

def calculate_accuracy(X_test, y_test, predict_fn, num_samples=1000):
    correct_predictions = 0
    predictions = []
    
    for i in range(min(num_samples, len(X_test))):
        true_label = y_test[i]
        predicted_label = predict_fn(X_test[i])
        predictions.append(predicted_label)
        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = (correct_predictions / min(num_samples, len(X_test))) * 100
    print(f"Accuracy on the test set: {accuracy:.2f}%")
    return predictions

def display_predictions(X_test, predictions, num_samples=8):
    fig, axs = plt.subplots(2, 4, figsize=(10, 4))
    for i, ax in enumerate(axs.flatten()):
        if i >= num_samples:
            break
        ax.imshow(X_test[i], cmap='gray')
        ax.set_title(f"Pred={predictions[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
