""" CITS4402 COMPUTER VISION - PROJECT
    HOLOGRAPHIC ACQUISITION RIG AUTOMATIC CALIBRATION
    ERWIN BAUERNSCHMITT, ALIAN HAIDAR, LUKE KIRKBY
    22964301, 22900426, 22885101
"""

import math
import numpy as np
import cv2
import os
from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QPixmap, QImage
import json


class HexaTargetIdentifier:
    """ Uniquely identifies and labels HexTargets in the given image,
        Returns a list of HexTargets with details and a labelled image.
    """
    def __init__(self, image):
        self.image = image
        self.HexaTargets, self.labelled_image = self.run()


    def colour_segment_image(self, requested_colour): 
            """ Segments image for requested colour.
                (requested_colour = "blue", "red", or "green")
            """ 
            image = self.image.copy()
            hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Defines threshold values where colours = {colour: [lower_hsv_threshold, upper_hsv_threshold]}
            # colours = {"blue": [[90,100,60],[130,255,255]], "red": [[169,100,60],[179,255,255]], "green": [[40,45,40],[90,255,255]]}
            colours = {"blue1":  [[ 98, 29,  0],[141,255,113]], 
                        "blue2":  [[ 98, 97, 96],[130,255,255]],
                        "red1":   [[150, 35,  0],[179,255,150]], 
                        "red2":   [[150,100,100],[179,255,223]],
                        "red3":   [[  0, 35, 60],[ 10,255,223]],
                        "green1": [[ 37,  0,  0],[102,160,100]],
                        "green2": [[ 37, 42,101],[ 91,255,168]]}

            # Creates thresholding mask.
            lower_hsv_threshold = np.array(colours[requested_colour][0])
            upper_hsv_threshold = np.array(colours[requested_colour][1])
            thresholding_mask = cv2.inRange(hsvimage, lower_hsv_threshold, upper_hsv_threshold)

            # Applies thresholding mask and binarizes result.
            segmented_image = cv2.bitwise_and(hsvimage, hsvimage, mask = thresholding_mask)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
            ret, segmented_image = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY)

            return(segmented_image)
    

    def filter_components(self, red_segments, green_segments, blue_segments):
        """ Performs connected components analysis on a binarized image,
            Thresholds the components by their area,
            Returns all the dot-sized objects with their info.
        """

        min_area = 2
        max_area = 165

        red_components = []
        (red_numLabels, red_labels, red_stats, red_centroids) = cv2.connectedComponentsWithStats(red_segments, connectivity = 8, ltype = cv2.CV_32S)

        green_components = []
        (green_numLabels, green_labels, green_stats, green_centroids) = cv2.connectedComponentsWithStats(green_segments, connectivity = 8, ltype = cv2.CV_32S)

        blue_components = []
        (blue_numLabels, blue_labels, blue_stats, blue_centroids) = cv2.connectedComponentsWithStats(blue_segments, connectivity = 8, ltype = cv2.CV_32S)




        for i in range(1, blue_numLabels): #starts at 1 to exlude the background
            # Extract the connected component stats for the current label
            x = blue_stats[i, cv2.CC_STAT_LEFT]
            y = blue_stats[i, cv2.CC_STAT_TOP]
            w = blue_stats[i, cv2.CC_STAT_WIDTH]
            h = blue_stats[i, cv2.CC_STAT_HEIGHT]
            area = blue_stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = blue_centroids[i]

            if (min_area <= area <= max_area) and (0.45 < w / h < 1.35): 
                blue_component = [x,y,w,h,cX,cY,area,[0,0,255]]
                blue_components.append(blue_component)

    
        for i in range(1, red_numLabels): #starts at 1 to exlude the background
            # Extract the connected component stats for the current label
            x = red_stats[i, cv2.CC_STAT_LEFT]
            y = red_stats[i, cv2.CC_STAT_TOP]
            w = red_stats[i, cv2.CC_STAT_WIDTH]
            h = red_stats[i, cv2.CC_STAT_HEIGHT]
            area = red_stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = red_centroids[i]

            if (min_area <= area <= max_area) and (0.3 < w / h < 1.7): 
                red_component = [x,y,w,h,cX,cY,area,[255,0,0]]
                red_components.append(red_component)


        for i in range(1, green_numLabels): #starts at 1 to exlude the background
            # Extract the connected component stats for the current label
            x = green_stats[i, cv2.CC_STAT_LEFT]
            y = green_stats[i, cv2.CC_STAT_TOP]
            w = green_stats[i, cv2.CC_STAT_WIDTH]
            h = green_stats[i, cv2.CC_STAT_HEIGHT]
            area = green_stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = green_centroids[i]

            if (min_area <= area <= max_area) and (0.3 < w / h < 1.7): 
                green_component = [x,y,w,h,cX,cY,area,[0,255,0]]
                green_components.append(green_component)


        blue_refined = []
        # Only keeps blue components if they are the largest of their neighbours
        for index in range(len(blue_components)):
            current_component = blue_components.pop(index)
            nearby_components = []
            for component in blue_components:
                if np.absolute(component[4] - current_component[4]) <= 30:
                    if np.absolute(component[5] - current_component[5]) <= 40:
                        nearby_components.append(component)

            # Append current_component to blue_refined if its area is the greatest
            if all(current_component[6] >= component[6] for component in nearby_components):
                blue_refined.append(current_component)

            # Insert the current component back into blue_components at its correct index
            blue_components.insert(index, current_component)
        blue_components = blue_refined






        green_refined = []
        # Only keeps green components if they are close to a blue component and similar in size to the blue component
        for green_component in green_components:
            for blue_component in blue_components:
                if np.absolute(blue_component[4] - green_component[4]) <= 30:
                    if 0 < green_component[5] - blue_component[5] <= 60:
                        if 0.6 <= green_component[6] / blue_component[6] <= 1.5:
                            green_refined.append(green_component)
                            break  # Breaks the inner loop and moves to the next green component if a close blue component is found
        green_components = green_refined

        red_refined = []
        # Only keeps red components if they are close to a blue component and similar in size to the blue component
        for red_component in red_components:
            for blue_component in blue_components:
                if np.absolute(blue_component[4] - red_component[4]) <= 30:
                    if 0 < red_component[5] - blue_component[5] <= 60:
                        if 0.7 < red_component[6] / blue_component[6] < 1.6:
                            red_refined.append(red_component)
                            break  # Breaks the inner loop and moves to the next red component if a close blue component is found
        red_components = red_refined




        blue_refined = []
        # Only keeps blue components if they have at least six neighbours of similar size.
        for blue_component in blue_components:
            potential_HexaTarget = [blue_component]

            for green_component in green_components:
                if np.absolute(green_component[4] - blue_component[4]) <= 30:
                    if 0 < green_component[5] - blue_component[5] <= 60:
                        if 0 < green_component[6] / blue_component[6] < 2:
                            potential_HexaTarget.append(green_component)

            for red_component in red_components:
                if np.absolute(red_component[4] - blue_component[4]) <= 30:
                    if 0 < red_component[5] - blue_component[5] <= 60:
                        if 0.5 < red_component[6] / blue_component[6] < 1.8:
                            potential_HexaTarget.append(red_component)

            if (len(potential_HexaTarget) >= 6):
                blue_refined.append(blue_component)
        blue_components = blue_refined

        return red_components, green_components, blue_components
    

    def group_components_into_HexaTargets(self, all_dots):
        """ Groups components into HexaTargets spatially.
        """ 
        red_dots = all_dots[0]
        green_dots = all_dots[1]
        blue_dots = all_dots[2]

        HexaTargets = []

        # Groups dots into HexaTargets by spatial proximity.
        for blue_dot in blue_dots:
            new_HexaTarget = [blue_dot]

            for green_dot in green_dots:
                if np.absolute(green_dot[4] - blue_dot[4]) <= 40:
                    if 0 < green_dot[5] - blue_dot[5] <= 60:
                        new_HexaTarget.append(green_dot)

            for red_dot in red_dots:
                if np.absolute(red_dot[4] - blue_dot[4]) <= 40:
                    if 0 < red_dot[5] - blue_dot[5] <= 60:
                        new_HexaTarget.append(red_dot)

            if (len(new_HexaTarget) == 6):
                HexaTargets.append(new_HexaTarget)

            elif (len(new_HexaTarget) > 6):
                # If more than 5 dots are close to the blue dot, compute distances and keep only the 5 closest ones
                # Compute Euclidean distance from the blue dot to each other dot in new_HexaTarget
                distances = [((dot[4] - blue_dot[4]) ** 2 + (dot[5] - blue_dot[5]) ** 2) ** 0.5 for dot in new_HexaTarget]
                # Sort new_HexaTarget by these distances
                new_HexaTarget = [x for _, x in sorted(zip(distances, new_HexaTarget))]
                # Keep only the blue dot and the 5 closest other dots
                new_HexaTarget = new_HexaTarget[:6]
                HexaTargets.append(new_HexaTarget)
        
        # Final check for HexaTargets
        HexaTargets_refined = []
        for HexaTarget in HexaTargets:
            # Compute mean centroid
            mean_cX = sum(dot[4] for dot in HexaTarget) / 6
            mean_cY = sum(dot[5] for dot in HexaTarget) / 6

            # Compute distances from mean centroid to each dot's centroid
            distances_to_mean = [((dot[4] - mean_cX) ** 2 + (dot[5] - mean_cY) ** 2) ** 0.5 for dot in HexaTarget]

            # Compute mean distance
            mean_distance = sum(distances_to_mean) / 6

            # Check if all dots are within an acceptable range of the mean distance
            if all(0.7 * mean_distance <= distance <= 1.3 * mean_distance for distance in distances_to_mean):
                HexaTargets_refined.append(HexaTarget)

        return HexaTargets_refined



    def group_dots_into_HexaTargets(self, all_dots):
        """ Appends colours to the dots,
            Groups them into HexaTargets spatially.
        """ 
        blue_dots = all_dots[0]
        green_dots = all_dots[1]
        red_dots = all_dots[2]
        HexaTargets = []

        # Appends colours to the dots.
        for blue_dot in blue_dots:
            blue_dot.append([0, 0, 255])
        for green_dot in green_dots:
            green_dot.append([0, 255, 0])
        for red_dot in red_dots:
            red_dot.append([255, 0, 0])

        # Groups dots into HexaTargets by spatial proximity.
        for blue_dot in blue_dots:
            new_HexaTarget = [blue_dot]

            for green_dot in green_dots:
                if np.absolute(green_dot[4] - blue_dot[4]) <= 40:
                    if 0 < green_dot[5] - blue_dot[5] <= 60:
                        new_HexaTarget.append(green_dot)

            for red_dot in red_dots:
                if np.absolute(red_dot[4] - blue_dot[4]) <= 40:
                    if 0 < red_dot[5] - blue_dot[5] <= 60:
                        new_HexaTarget.append(red_dot)

            if (len(new_HexaTarget) == 6):
                HexaTargets.append(new_HexaTarget)

        return HexaTargets


    def order_HexaTarget_dots(self, HexaTargets):
        """ Orders each HexaTarget's dots in clockwise direction.
        """
        ordered_HexaTargets = []

        for HexaTarget in HexaTargets:
            HexaTarget = HexaTarget.copy()

            blue = HexaTarget[0]
            blue_x_centroid = blue[4]
            blue_y_centroid = blue[5]
            HexaTarget.pop(0)

            # calculate angles relative to blue
            for point in HexaTarget:
                dx = point[4] - blue_x_centroid
                dy = point[5] - blue_y_centroid
                angle = math.atan2(dy, dx)
                point.append(angle)

            # sort the points in the HexaTarget by their angle
            HexaTarget.sort(key=lambda point: point[-1])

            # insert the blue point back into its original position
            HexaTarget.insert(0, blue)
            
            # remove angle from points data
            for point in HexaTarget[1:]:
                point.pop(-1)

            ordered_HexaTargets.append(HexaTarget)

        return ordered_HexaTargets


    def uniquely_identify_HexaTargets(self, ordered_HexaTargets):
        """ Appends a string to each HexaTarget, uniquely identifying its colour sequence.
        """
        uniquely_identified_HexaTargets = []

        for HexaTarget in ordered_HexaTargets:
            colour_sequence = ""
            for dot in HexaTarget[1:]:
                if dot[-1] == [0, 0, 255]:
                    colour_sequence += "B"
                if dot[-1] == [0, 255, 0]:
                    colour_sequence += "G"
                if dot[-1] == [255, 0, 0]:
                    colour_sequence += "R"
            HexaTarget.append(colour_sequence)
            uniquely_identified_HexaTargets.append(HexaTarget)
        
        return uniquely_identified_HexaTargets


    def uniquely_label_dots(self, uniquely_identified_HexaTargets):
        """ Draws coloured rectangles around each dot,
            Labels each dot with dot's number and HexaTarget's colour sequence.
        """   
        labelled_image = self.image.copy()
        labelled_image = cv2.cvtColor(labelled_image, cv2.COLOR_BGR2RGB)

        # For each HexaTarget:
        for HexaTarget in uniquely_identified_HexaTargets:
            # For each dot:
            for i, dot in enumerate(HexaTarget[:-1], start=1):
                # Draws rectangles around each dot.            
                labelled_image = cv2.rectangle(
                    img = labelled_image,
                    pt1 = (dot[0], dot[1]),
                    pt2 = (dot[0] + dot[2], dot[1] + dot[3]),
                    color = dot[-1],
                    thickness = 1
                )

                # Draws centroid point for each dot.
                cv2.circle(
                    img = labelled_image,
                    center = (round(dot[4]), round(dot[5])),  # centroid coordinates
                    radius = 1, 
                    color = dot[-1],
                    thickness = -1 
                )

                # Writes unique identifier above each dot.
                cv2.putText(
                    img = labelled_image,
                    text = f"{HexaTarget[-1]}_{i}",
                    org = (dot[0] - 15, dot[1]),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.3,
                    color = dot[-1],
                    thickness = 1
                )
            
        return labelled_image
            

        



    def run(self):
        """ Executes the HexTarget identification and labelling.
        """
        red_segments1 = self.colour_segment_image("red1")
        red_segments2 = self.colour_segment_image("red2")
        red_segments3 = self.colour_segment_image("red3")
        red_segments = cv2.bitwise_or(cv2.bitwise_or(red_segments1, red_segments2), red_segments3)
        green_segments1 = self.colour_segment_image("green1")
        green_segments2 = self.colour_segment_image("green2")
        green_segments = cv2.bitwise_or(green_segments1, green_segments2)
        blue_segments1 = self.colour_segment_image("blue1")
        blue_segments2 = self.colour_segment_image("blue2")
        blue_segments = cv2.bitwise_or(blue_segments1, blue_segments2)

        red_components, green_components, blue_components = self.filter_components(red_segments, green_segments, blue_segments)
        all_components = [red_components, green_components, blue_components]
        HexaTargets = self.group_components_into_HexaTargets(all_components)
        ordered_HexaTargets = self.order_HexaTarget_dots(HexaTargets)
        uniquely_identified_HexaTargets = self.uniquely_identify_HexaTargets(ordered_HexaTargets)
        labelled_image = self.uniquely_label_dots(uniquely_identified_HexaTargets)

        return uniquely_identified_HexaTargets, labelled_image


