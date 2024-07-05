import cv2
from matplotlib import pyplot as plt

img = cv2.imread('foto.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

