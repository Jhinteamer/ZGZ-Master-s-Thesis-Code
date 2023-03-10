import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(-9, 9, 0.1)
y = []
for x1 in x:
    # y1 = math.exp(x)
    y1 = 1 / (1 + math.exp(-x1))
    y.append(y1)
plt.plot(x, y, label = "Sigmoid")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-9, 9)
plt.ylim(-0.1, 1.1)
plt.legend()
plt.savefig("./sigmoid.png")
# plt.show()