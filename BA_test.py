import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

sns.set(style="darkgrid")

cev_alpha = 0.5
vec_alpha = 0.5
lda = 0.5
p = 0.3

x1 = np.arange(0.0, 1.001, 0.001)
x2 = np.arange(0.0, 1.001, 0.001)
A = np.zeros((1001,1001))

for i in np.arange(len(x1)):
    for j in np.arange(len(x2)):
        y1 = cev_alpha * x2[j] * x1[i] + x1[i] - (1 - lda) * ((1 / p) - 1)
        y2 = vec_alpha * x1[i] * x2[j] + x2[j] - lda * ((1 / p) - 1)
        if y1 > 0 and y2 > 0:
            A[i,j] = 1

plt.matshow(A)
plt.colorbar()
