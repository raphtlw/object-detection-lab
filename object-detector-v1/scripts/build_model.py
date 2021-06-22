import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

classifier_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential(
    [hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))]
)

grace_hopper = tf.keras.utils.get_file(
    "image.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
)
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper) / 255.0
print(grace_hopper.shape)
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)  # type: ignore
predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)
labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
)
imagenet_labels = np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis("off")
predicted_class_name = imagenet_labels[predicted_class]
prediction_display_output = plt.title("Prediction: " + predicted_class_name.title())
print(prediction_display_output)
# plt.show()
