

import tensorflow as tf
import time

# Перевірка наявності GPU
device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError("GPU device not found")
print(f"Found GPU at: {device_name}")

# Генерація даних
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Створення простої моделі
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Тренування моделі та вимірювання часу
start_time = time.time()
model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=2)
end_time = time.time()

print(f"Training time on GPU: {end_time - start_time:.2f} seconds")
#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#%%
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        details = tf.config.experimental.get_memory_info('GPU:0')
        print(f"Available memory: {details['current']} bytes")
