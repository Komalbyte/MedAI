import json
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes=5):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None)
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.2),
    ])
    inputs = layers.Input(shape=(224, 224, 3))
    x = augment(inputs)
    x = layers.Rescaling(1/127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

def load_medai():
    class_names = json.load(open("class_names.json"))
    model = build_model(len(class_names))
    model.load_weights("medai.weights.h5")
    return model, class_names

if __name__ == "__main__":
    model, class_names = load_medai()
    print("WEIGHTS OK", model.output_shape)
    print(class_names)
