import cv2
import imutils


class ColorDescriptor:
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins,
                            [0, 180, 0, 256, 0, 256])
        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()
        # return the histogram
        return hist.tolist()
