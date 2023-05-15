import math
import numpy as np
import cv2


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

            # Defines threshold values where colours = {colour: [lower_hsv_threshold, upper_hsv_threshold]}.
            colours = {"blue": [[90,100,60],[130,255,255]], "red": [[169,100,60],[179,255,255]], "green": [[40,45,40],[90,255,255]]}

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


    def identify_dots(self, segmented_image):
        """ Performs connected components analysis on a binarized image,
            Thresholds the components by their area,
            Returns all the dot-sized objects with their info.
        """
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(segmented_image, cv2.CV_32S)

        dots = []

        for i in range(1, numLabels): #starts at 1 to exlude the background
            # Extract the connected component stats for the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            if 50 <= area <= 200: 
                dot = [x,y,w,h,cX,cY,area]
                dots.append(dot)

        return dots


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
                if np.absolute(green_dot[4] - blue_dot[4]) <= 75:
                    if np.absolute(green_dot[5] - blue_dot[5]) <= 75:
                        new_HexaTarget.append(green_dot)

            for red_dot in red_dots:
                if np.absolute(red_dot[4] - blue_dot[4]) <= 75:
                    if np.absolute(red_dot[5] - blue_dot[5]) <= 75:
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
                    text = f"HexaTarget_{HexaTarget[-1]}_{i}",
                    org = (dot[0] - 45, dot[1]),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.3,
                    color = dot[-1],
                    thickness = 1
                )
            
        return labelled_image
            
    
    def run(self):
        """ Executes the HexTarget identification and labelling.
        """
        red_segments = self.colour_segment_image("red")
        green_segments = self.colour_segment_image("green")
        blue_segments = self.colour_segment_image("blue")

        red_dots = self.identify_dots(red_segments)
        green_dots = self.identify_dots(green_segments)
        blue_dots = self.identify_dots(blue_segments)
        all_dots = [blue_dots, green_dots, red_dots]

        HexaTargets = self.group_dots_into_HexaTargets(all_dots)
        ordered_HexaTargets = self.order_HexaTarget_dots(HexaTargets)
        uniquely_identified_HexaTargets = self.uniquely_identify_HexaTargets(ordered_HexaTargets)
        labelled_image = self.uniquely_label_dots(uniquely_identified_HexaTargets)

        return uniquely_identified_HexaTargets, labelled_image


