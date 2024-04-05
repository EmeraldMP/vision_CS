import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, color
from skimage import graph
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from matplotlib import pyplot
from numpy import unique
from numpy import where
import numpy as np

def human_shape(img_path, display_img=False):
    # Create an instance of the PoseDetector class
    pose = PoseDetector(img_path, display=display_img)
    # Run the pose detector
    img, mask = pose.run()

    return img, mask

class PoseDetector:
        
    def __init__(self, img_path, desired_width=400, display=True):
        self.display = display  
        self.desired_width = desired_width
        self.resize_shape = None
        self.img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.img_resized = None
    
    def display_image(self, img, title='Image', gray=False):
        if self.display:
            plt.imshow(img)
            # If the image is in grayscale, display it in grayscale
            if gray:
                plt.gray()

            plt.title(title)
            plt.axis('off')
            plt.show()
    
    def resize_image(self, desired_width):
        aspect_ratio = self.img.shape[1] / self.img.shape[0]
        desired_height = int(desired_width / aspect_ratio)
        self.img_resized = cv2.resize(self.img, (desired_width, desired_height))
        return self.img_resized.copy()

    def gaussian_blur(self, img, kernel_size=5, sigma=1.4):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma, sigma, cv2.BORDER_DEFAULT)
    
    
    def remove_background(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blk_thresh = gray.mean()  
        _, threshold_img = cv2.threshold(gray, blk_thresh, 255, cv2.THRESH_BINARY_INV)

        # Display the thresholded image
        self.display_image(threshold_img, 'Threshold Image', gray=True)

        # Apply mask
        masked_img = cv2.bitwise_and(img, img, mask=threshold_img)
        # masked_img = cv2.bitwise_and(self.img_resized, self.img_resized, mask=threshold_img)

        return masked_img

    def greates_contour(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use thresholding to create a binary image
        _, thresh = cv2.threshold(gray, gray.mean(), 255, cv2.THRESH_BINARY_INV)
        self.display_image(thresh, 'Threshold Image 2', gray=True)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort the contours by area and get the largest one
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the largest contour
        mask = np.zeros_like(gray)

        cv2.drawContours(mask, [largest_contour], 0, color=255, thickness=-1)

        # Create an image only containing the person
        person = np.zeros_like(img)
        person[mask == 255] = img[mask == 255]

        return person
    
    def cluster_pixels_DBSCAN_gray(self, img, eps=3, min_samples=2):

        # Create a binary mask: 1 for the person, 0 for the background
        binary_mask = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) != 0, 1, 0)
        y_shape, x_shape = binary_mask.shape

        # Get the coordinates of the person
        pixels = []
        for i in range(y_shape):
            for j in range(x_shape):
                if binary_mask[i, j] == 1:
                    pixels.append([j, y_shape - i]) # x, y
        pixels = np.array(pixels)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        yhat = model.fit_predict(pixels)
        clusters = unique(yhat)

        row_ix = where(yhat == 0)
        points = [pixels[row_ix, 0][0], pixels[row_ix, 1][0]]

        final_mask = np.zeros_like(binary_mask)

                # put zeros in the mask where the person is not
        for i in range(len(points[1])):
            final_mask[int(y_shape - points[1][i]), int(points[0][i])] = 255

        self.display_image(final_mask, 'Cluster mask', gray=True)

        # Apply the mask to the image
        final_person = np.zeros_like(img)
        final_person[final_mask == 255] = self.img_resized[final_mask == 255]

        return final_person, final_mask


    def run(self):
        # Display image
        self.display_image(self.img, 'Original Image')

        # Resizing the image
        img = self.resize_image(self.desired_width)
        self.display_image(img, 'Resized Image')

        # Application of Gaussian Blur
        img = self.gaussian_blur(img, kernel_size=15, sigma=10)
        self.display_image(img, 'Gaussian Blur Image')

        # # Remove Background
        img = self.remove_background(img)
        self.display_image(img, 'Background Removed Image')

        # find contours
        img = self.greates_contour(img)
        self.display_image(img, 'Greates Contour Image')

        # Cluster pixels using DBSCAN just gray
        img, mask = self.cluster_pixels_DBSCAN_gray(img, eps=4, min_samples=2)
        self.display_image(img, 'Clustered Pixels Image')
        self.display_image(mask, 'Final Clustered Pixels Mask', gray=True)

        return self.img_resized, mask

if __name__ == '__main__':
    path_img = 'data/image4.jpg'
    pose = PoseDetector(path_img, display=True)
    img, mask = pose.run()
