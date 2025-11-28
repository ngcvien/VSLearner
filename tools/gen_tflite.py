import tensorflow as tf

# Load model
model = tf.keras.models.load_model('../models/sign_model.h5')

# Convert sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Lưu lại
with open('../models/sign_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Đã tạo sign_model.tflite")
