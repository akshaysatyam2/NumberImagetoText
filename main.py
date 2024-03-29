import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model("ResNet50custom.model")

test_data = "dataset/1/capitalize_1.png"

# Read and preprocess the input image
img = cv2.imread(test_data)
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Make a prediction
prediction = model.predict(img)

# Print the predicted number
print(f"Number: {np.argmax(prediction)}")

# Display the image
cv2.imshow("Image", cv2.imread(test_data))
cv2.waitKey(0)

# ## For MINST model
#
# import tensorflow as tf
# import cv2
# import numpy as np
#
# model = tf.keras.models.load_model("num.model")
#
# img = cv2.imread("Untitled.png")[:, :, 0]
# img = cv2.resize(img, (28, 28))
#
# img = np.invert(np.array([img]))
#
# img = img.reshape(1, 28, 28)
#
# img = img / 255.0
#
# # Make a prediction
# prediction = model.predict(img)
#
# # Print the predicted number
# print(f"Number: {np.argmax(prediction)}")
# img = cv2.imread("Untitled.png")
# cv2.imshow("Image", img)
# cv2.waitKey(0)
