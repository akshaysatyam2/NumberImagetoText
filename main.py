import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("num.model")

img = cv2.imread("Untitled.png")[:, :, 0]
img = cv2.resize(img, (28, 28))

img = np.invert(np.array([img]))

img = img.reshape(1, 28, 28)

img = img / 255.0

# Make a prediction
prediction = model.predict(img)

# Print the predicted number
print(f"Number: {np.argmax(prediction)}")
img = cv2.imread("Untitled.png")
cv2.imshow("Image", img)
cv2.waitKey(0)
