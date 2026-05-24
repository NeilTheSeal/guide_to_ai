import matplotlib.pyplot as plt
import numpy as np

examples = np.array(
    [
        [[1, 1, 1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1]],
        [[-1, 1, 1, 1], [-1, -1, 1, -1], [-1, -1, 1, -1], [-1, -1, 1, -1]],
        [[-1, -1, 1, -1], [-1, -1, 1, -1], [1, -1, 1, -1], [1, 1, 1, -1]],
        [[-1, -1, -1, 1], [-1, -1, -1, 1], [-1, 1, -1, 1], [-1, 1, 1, 1]],
    ]
)

fig = plt.figure(0, (6, 6))
for i, example in enumerate(examples):
    fig.add_subplot(2, 2, i + 1)
    plt.imshow(example)

# plt.show()

y = np.array([1, 1, -1, -1])

X = np.hstack((examples.reshape(-1, 16), np.ones((len(y), 1))))

# print(X.shape)
# print(y.shape)
# print(X)

w = np.zeros(17)
lr = 1.0

for i in range(1, 10):
    yhat = np.dot(X[i % len(y)], w)
    if y[i % len(y)] > 0 <= yhat:
        print(f"output is {yhat} but we want it to be {y[i % len(y)]}, updating weights")
        w = w + lr * X[i % len(y)]
    elif yhat > 0 <= y[i % len(y)]:
        print(f"output is {yhat} but we want it to be {y[i % len(y)]}, updating weights")
        w = w - lr * X[i % len(y)]
    else:
        print(
            f"output is {yhat}, which has the same sign as our target {y[i % len(y)]}, machine is correct. "
            "not updating weights"
        )
