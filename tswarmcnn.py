import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import secrets
from multiprocessing import Process, Queue
from opti_algo.tswarm import TSwarm

(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


def get_model():
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3))
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model


def wrapped_fitter(learning_rate, batch_size, epoches, q):
    model = get_model()
    nn_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=nn_optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    seed = secrets.randbits(128)
    rng1 = np.random.default_rng(seed)
    rng1.shuffle(train_images)
    rng2 = np.random.default_rng(seed)
    rng2.shuffle(train_labels)
    validation_images = train_images[45000:]
    validation_labels = train_labels[45000:]
    history = model.fit(
        train_images[0:45000],
        train_labels[0:45000],
        batch_size=batch_size,
        epochs=epoches,
        validation_data=(validation_images, validation_labels),
        verbose=1,
    )
    q.put(1 - history.history["val_accuracy"][-1])


def secondary_wrapper(args):
    argument = args.copy()
    argument.append(q)
    argument[1] = int(args[1])
    argument[2] = int(args[2])
    p = Process(target=wrapped_fitter, args=argument)
    p.start()
    result = q.get()
    p.join()
    return result


if __name__ == "__main__":
    q = Queue()
    tswarm_algo = TSwarm(
        secondary_wrapper,
        3,
        [(0.0005, 0.05), (16, 512), (8, 16)],
        args=(1, 3, 0.4, 1, 5),
    )
    result = tswarm_algo.optimize()
    tswarm_algo.visualize()
    print(result)


# history = model.fit(
#     train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)
# )

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
