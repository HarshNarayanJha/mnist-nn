# %% [md]
"""
### Importing Libraries
"""

# %%
import keras
import matplotlib.pyplot as plt
import numpy as np

# %%
X_train_full: np.ndarray[int, np.dtype[np.float64]]
y_train_full: np.ndarray[int, np.dtype[np.float64]]
X_test: np.ndarray[int, np.dtype[np.float64]]
y_test: np.ndarray[int, np.dtype[np.float64]]

mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# %%
X_train_full.shape
# %%
X_train_full[0]

# %%
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        axes[i, j].imshow(X_train_full[np.random.randint(0, X_train_full.shape[0] - 1)], cmap=plt.get_cmap("gray"))

plt.show()

# %%
X_val, X_train = X_train_full[:5000] / 255, X_train_full[5000:] / 255
y_val, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255

# %%
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
n = int(np.random.random() * y_train.shape[0])

plt.imshow(X_train[n], cmap=plt.get_cmap("gray"))
plt.show()
print(class_names[y_train[n]])
