import tensorflow as tf

# Define .keras model to convert, as well as the .tflite output
keras_model_path = 'cnn_final_model.keras'
output_tflite_path = 'cnn_final_model.tflite'

# Convert the Keras model to a TensorFlow Lite model
model = tf.keras.models.load_model(keras_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(output_tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"New model saved to {output_tflite_path}")