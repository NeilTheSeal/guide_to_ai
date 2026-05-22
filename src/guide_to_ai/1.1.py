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

plt.show()
