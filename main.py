import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

class ImagePreprocessing:
    def __init__(self):
        self.train_dir = "/Users/Akhil/PycharmProjects/Blood-Cell-Image-Segmentation/dataset-master/JPEGImages"

    def read_images(self):
        images = []
        for filename in os.listdir(self.train_dir):
            img = cv2.imread(os.path.join(self.train_dir, filename))
            if img is not None:
                images.append(img)

        return images


x = ImagePreprocessing()
images = x.read_images()
plt.imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
plt.show()

