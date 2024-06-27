import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fast')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"number of training examples: {m}")

# mm is the number pf training examples with len() python function
m = len(x_train)
print(f"Number of training examples is: {m}")

# x_i is the i-th training example
i = 1

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Plot the training data
plt.scatter(x_train, y_train, marker='x', color='r')
# set the title
plt.title("Housing Prices")
# set the x-axis label
plt.xlabel('Size (1000 sqft)')
# Set the y-axis label
plt.ylabel('Price (1000 $)')
plt.show()
