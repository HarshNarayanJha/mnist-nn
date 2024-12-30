# %% [md]
"""
### Importing Libraries
"""

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

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

# %% [md]
"""
### Enter Easy Neural Networks!
"""

# %%
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# sigmoid: probabilities produces are independent (sum to more than 1)
# softmax: probabilities produces are dependent, they sum to 1

# %%
model.summary()

# %%
model.layers

# %%
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# %%
X_train.shape
# %%
history: keras.callbacks.History = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# %%
pd.DataFrame(history.history).plot(figsize=(15, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# %%
model.evaluate(X_test, y_test)
# %%
y_prob: np.ndarray = model.predict(X_test)
y_classes = y_prob.argmax(axis=-1)
print(y_classes)
# %%
confusion_matrix = tf.math.confusion_matrix(y_test, y_classes)

# %%
axs = sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")

axs.set_xlabel("Predicted Labels")
axs.set_ylabel("True Labels")
axs.set_title("Confusion Matrix")
axs.xaxis.set_ticklabels(class_names)
axs.yaxis.set_ticklabels(class_names)
axs.figure.set_size_inches(12, 10)

plt.show()

# %%
# Custom Data Prediction

from PIL import Image  # noqa: E402

imgs = [Image.open(x).convert("L") for x in ("./2.jpg", "./3.jpg", "./5.jpg")]
imgs = np.array([np.array(x) for x in imgs])
pred = model.predict(imgs)
preds = pred.argmax(axis=-1)
for img, pred in zip(imgs, preds):
    plt.imshow(img, cmap=plt.get_cmap("gray"))
    plt.show()
    print("Predicted Class:", class_names[pred])
# %%
model.save("mnist_digit.keras")
model.save_weights("mnist_digit.weights.h5")
