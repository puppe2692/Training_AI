import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fast')


def compute_model_output(x, w, b):
    m = len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

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

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b,label='Our Prediction')
# Plot the training data
plt.scatter(x_train, y_train, marker='x', color='r', label='Actual Values')
# set the title
plt.title("Housing Prices")
# set the x-axis label
plt.xlabel('Size (1000 sqft)')
# Set the y-axis label
plt.ylabel('Price (1000 $)')
plt.legend()
plt.show()

w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")