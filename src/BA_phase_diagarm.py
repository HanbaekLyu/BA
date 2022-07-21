import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

sns.set(style="darkgrid")

x = np.arange(0.0, 1.001, 0.001)
y = np.maximum((1-2*x)/(2-2*x), (2*x-1)/(2*x))
y1 = np.maximum(y, 1/5)
y2 = np.maximum(y, 1/4)
y3 = np.maximum(x/(1+x), (1-x)/(2-x))
y4 = np.ones(len(x))

fig, ax = plt.subplots()
pal = sns.color_palette("Set1")
pall =["lightskyblue", "azure", "#34495e", "#2ecc71"]
plt.stackplot(x, y1, y2-y1, y3-y2, y4-y3, colors=pall, alpha=0.4)
plt.axis([0, 1, 0, 1])
ax.xaxis.set_major_locator(ticker.MultipleLocator(1/6))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1/6))
ax.set_aspect('equal')
plt.show()
