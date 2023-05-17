import math
import numpy as np
import cv2


class ImageSegmenter:
    """ Segments the image with the given thresholds.
    """
    def __init__(self, image, colour_thresholds):
        self.image = image
        self.colour_thresholds = colour_thresholds
        self.segmented_image = self.run()


    def colour_segment_image(self, requested_colour): 
            """ Segments image for requested colour.
                (requested_colour = "blue", "red", or "green")
            """ 
            image = self.image.copy()
            hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Defines threshold values where colours = {colour: [lower_hsv_threshold, upper_hsv_threshold]}
            # colours = {"blue": [[90,100,60],[130,255,255]], "red": [[169,100,60],[179,255,255]], "green": [[40,45,40],[90,255,255]]}
            # colours = {"blue": [[90,20,40],[130,250,200]], "red1": [[160,20,40],[179,250,250]], "red2": [[0,20,40],[10,250,250]], "green": [[40,20,40],[90,250,250]]}

            # Creates thresholding mask.
            lower_hsv_threshold = np.array(self.colour_thresholds[requested_colour][0])
            upper_hsv_threshold = np.array(self.colour_thresholds[requested_colour][1])
            thresholding_mask = cv2.inRange(hsvimage, lower_hsv_threshold, upper_hsv_threshold)

            # Applies thresholding mask and binarizes result.
            segmented_image = cv2.bitwise_and(hsvimage, hsvimage, mask = thresholding_mask)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
            ret, segmented_image = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY)

            return(segmented_image)


    
    def run(self):
        """ Executes the HexTarget identification and labelling.
        """
        red_segments1 = self.colour_segment_image("red1")
        red_segments2 = self.colour_segment_image("red2")
        red_segments = red_segments1 + red_segments2
        green_segments = self.colour_segment_image("green")
        blue_segments = self.colour_segment_image("blue")

        # For testing:
        all_segments = red_segments + blue_segments + green_segments


        return all_segments


