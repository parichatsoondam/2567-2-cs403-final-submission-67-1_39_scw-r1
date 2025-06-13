import tensorflow as tf

model = tf.keras.models.load_model("mobilenet_best_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("mobilenet_model.tflite", "wb") as f:
    f.write(tflite_model)

print("mobilenet_model.tflite")
