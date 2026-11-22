import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# Create a test black square and save it
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
cv2.imwrite("test_patch.png", img)
print("Verification successful! Generated test_patch.png")