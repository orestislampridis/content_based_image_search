import cv2
import numpy as np


class ShapeDescriptor:
    def __init__(self, vector_size):
        # store the size of the vectors for our shape descriptors
        self.vector_size = vector_size

    def describe(self, image):
        try:
            # Using KAZE, cause SIFT, ORB and other was moved to additional module
            # which is adding addtional pain during install
            # sift = cv2.xfeatures2d.SIFT_create()
            # surf = cv2.xfeatures2d.SURF_create()
            kaze = cv2.KAZE_create()
            orb = cv2.ORB_create()

            # Find the image keypoints
            # sift_keypoints = sift.detect(image)
            #surf_keypoints = surf.detect(image)
            kaze_keypoints = kaze.detect(image)
            orb_keypoints = orb.detect(image)

            # Getting first 32 of them for each algorithm
            # Number of keypoints varies depending on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)
            # sift_keypoints = sorted(sift_keypoints, key=lambda x: -x.response)[:self.vector_size]
            #surf_keypoints = sorted(surf_keypoints, key=lambda x: -x.response)[:self.vector_size]
            kaze_keypoints = sorted(kaze_keypoints, key=lambda x: -x.response)[:self.vector_size]
            orb_keypoints = sorted(orb_keypoints, key=lambda x: -x.response)[:self.vector_size]

            # Compute descriptors vector
            # sift_keypoints, sift_descriptors = sift.compute(image, sift_keypoints)
            # surf_keypoints, surf_descriptors = surf.compute(image, surf_keypoints)
            kaze_keypoints, kaze_descriptors = kaze.compute(image, kaze_keypoints)
            orb_keypoints, orb_descriptors = orb.compute(image, orb_keypoints)

            # Flatten all of them in one big vector - our feature vector
            # sift_descriptors = sift_descriptors.flatten()
            # surf_descriptors = surf_descriptors.flatten()
            kaze_descriptors = kaze_descriptors.flatten()
            orb_descriptors = orb_descriptors.flatten()

            # Making descriptor of same vector size 32
            needed_size = (self.vector_size * 32)

            # If we have less the 32 descriptors then pad with zeroes
            if kaze_descriptors.size < needed_size:
                kaze_descriptors = np.concatenate([kaze_descriptors, np.zeros(needed_size - kaze_descriptors.size)])

            if orb_descriptors.size < needed_size:
                orb_descriptors = np.concatenate([orb_descriptors, np.zeros(needed_size - orb_descriptors.size)])

        except cv2.error as e:
            print('Error: ', e)
            return None

        return kaze_descriptors.tolist(), orb_descriptors.tolist()
