import math
import numpy as np
import cv2

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


    def colour_segment_image(self, requested_colour, image=None): 
            """ Segments image for requested colour.
                (requested_colour = "blue", "red", or "green")
            """ 
            if image is None:
                image = self.image.copy()

            else: 
                image = image.copy()

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


    # def uniquely_identify_HexaTargets(self, ordered_HexaTargets):
    #     """ Appends a string to each HexaTarget, uniquely identifying its colour sequence.
    #     """
    #     uniquely_identified_HexaTargets = []

    #     for HexaTarget in ordered_HexaTargets:
    #         colour_sequence = ""
    #         for dot in HexaTarget[1:]:
    #             if dot[-1] == [0, 0, 255]:
    #                 colour_sequence += "B"
    #             if dot[-1] == [0, 255, 0]:
    #                 colour_sequence += "G"
    #             if dot[-1] == [255, 0, 0]:
    #                 colour_sequence += "R"
    #         HexaTarget.append(colour_sequence)
    #         uniquely_identified_HexaTargets.append(HexaTarget)
        
    #     return uniquely_identified_HexaTargets
    


    def extract_dots(self, ordered_HexaTargets):
        image = self.image.copy()

        cropped_dots = []

        for HexaTarget in ordered_HexaTargets:
            for dot in HexaTarget:
                # Defines a box around the dot.
                scale_factor = 1.4

                if dot[2] <= 5 or dot[3] <= 5:
                    scale_factor = 1.8

                if dot[2] <= 3 or dot[3] <= 3:
                    scale_factor = 2.2

                box_x1 = round(dot[4] - dot[2] * 0.5 * scale_factor)
                box_x2 = round(dot[4] + dot[2] * 0.5 * scale_factor)
                box_y1 = round(dot[5] - dot[3] * 0.5 * scale_factor)
                box_y2 = round(dot[5] + dot[3] * 0.5 * scale_factor)

                # Extracts the dot as a new image and adds to list.
                cropped_dot = image[box_y1:box_y2, box_x1:box_x2]
                cropped_dots.append(cropped_dot)

                # Updates the HexaTarget pixel coordinates for the new origin.
                dot[0] = dot[0] - box_x1
                dot[1] = dot[1] - box_y1
                dot[4] = dot[4] - box_x1
                dot[5] = dot[5] - box_y1      

        ordered_HexaTargets = ordered_HexaTargets.copy()
        dots_info = []
        for HexaTarget in ordered_HexaTargets:
            dots_info += HexaTarget

        cropped_dots_with_info = list(zip(cropped_dots, dots_info))
        

        return cropped_dots, cropped_dots_with_info





    def label_dots(self, cropped_dots, ordered_HexaTargets=None, dots_info=None):
        """ Draws coloured rectangles around each dot,
            Labels each dot with dot's number and HexaTarget's colour sequence.
        """   
        dots = cropped_dots.copy()

        if dots_info is None:
            HexaTargets = ordered_HexaTargets.copy()
            dots_info = []
            for HexaTarget in HexaTargets:
                dots_info += HexaTarget



        cropped_dots_with_info = list(zip(dots, dots_info))

        resized_dots = []


        scale_factor = 16
        scaled_dots_info = []


        for (cropped_dot, dot_info) in cropped_dots_with_info:
            # Scale up the cropped dot image by 16 times
            resized_dot = cv2.resize(cropped_dot, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            resized_dots.append(resized_dot)

            scaled_dot_info = dot_info.copy()
            scaled_dot_info[0] = scaled_dot_info[0] * scale_factor
            scaled_dot_info[1] = scaled_dot_info[1] * scale_factor
            scaled_dot_info[2] = scaled_dot_info[2] * scale_factor
            scaled_dot_info[3] = scaled_dot_info[3] * scale_factor
            scaled_dot_info[4] = scaled_dot_info[4] * scale_factor
            scaled_dot_info[5] = scaled_dot_info[5] * scale_factor
            scaled_dot_info[6] = scaled_dot_info[2] * scale_factor * scale_factor

            scaled_dots_info.append(scaled_dot_info)

        resized_dots_with_info = list(zip(resized_dots, scaled_dots_info))

        labelled_dots = []

        # For each dot:
        for (resized_dot, dot_info) in resized_dots_with_info:
            resized_dot = cv2.cvtColor(resized_dot, cv2.COLOR_BGR2RGB)
            # Draws rectangles around the dot.   
            # print(resized_dot.shape)
            # print(dot_info)         
            resized_dot = cv2.rectangle(
                img = resized_dot,
                pt1 = (dot_info[0], dot_info[1]),
                pt2 = (dot_info[0] + dot_info[2], dot_info[1] + dot_info[3]),
                color = dot_info[-1],
                thickness = 1
            )

            # Draws centroid point for the dot.
            cv2.circle(
                img = resized_dot,
                center = (round(dot_info[4]), round(dot_info[5])),  # centroid coordinates
                radius = 1, 
                color = dot_info[-1],
                thickness = -1 
                )

            resized_dot = cv2.cvtColor(resized_dot, cv2.COLOR_RGB2BGR)
            labelled_dots.append(resized_dot)

                # # Writes unique identifier above each dot.
                # cv2.putText(
                #     img = labelled_image,
                #     text = f"{HexaTarget[-1]}_{i}",
                #     org = (dot[0] - 15, dot[1]),
                #     fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                #     fontScale = 0.3,
                #     color = dot[-1],
                #     thickness = 1
                # )   
            
        return labelled_dots



    def resegment_and_dilate(self, cropped_dots_with_info):
        cropped_dots_with_info = cropped_dots_with_info.copy()
        dilated_dots_with_info = []

        for (dot, info) in cropped_dots_with_info:
            if info[-1] == [255, 0, 0]:
                colour = "red"
            elif info[-1] == [0, 255, 0]:
                colour = "green"
            elif info[-1] == [0, 0, 255]:
                colour = "blue"

            if colour == "red":
                red_segments1 = self.colour_segment_image("red1", dot)
                red_segments2 = self.colour_segment_image("red2", dot)
                red_segments3 = self.colour_segment_image("red3", dot)
                red_segments = cv2.bitwise_or(cv2.bitwise_or(red_segments1, red_segments2), red_segments3)

                kernel = np.ones((3, 3))
                dilated_red_segments = cv2.dilate(red_segments, kernel, iterations=1)

                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2RGB)
                dilated_dot_with_info = (hsv_dot, dilated_red_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

            if colour == "green":
                green_segments1 = self.colour_segment_image("green1", dot)
                green_segments2 = self.colour_segment_image("green2", dot)
                green_segments = cv2.bitwise_or(green_segments1, green_segments2)

                kernel = np.ones((3, 3))
                dilated_green_segments = cv2.dilate(green_segments, kernel, iterations=1)

                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2RGB)
                dilated_dot_with_info = (hsv_dot, dilated_green_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

            if colour == "blue":
                blue_segments1 = self.colour_segment_image("blue1", dot)
                blue_segments2 = self.colour_segment_image("blue2", dot)
                blue_segments = cv2.bitwise_or(blue_segments1, blue_segments2)

                kernel = np.ones((3, 3))
                dilated_blue_segments = cv2.dilate(blue_segments, kernel, iterations=1)

                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2RGB)
                dilated_dot_with_info = (hsv_dot, dilated_blue_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

        updated_info = []

        for (dot, binarized_dot, info) in dilated_dots_with_info:
            dot = dot.astype(np.float32)
            normalized_dot = dot / np.linalg.norm(dot, axis=2, keepdims=True)
            r, g, b = cv2.split(normalized_dot)
            masked_r = cv2.bitwise_and(r, r, mask=binarized_dot)
            masked_g = cv2.bitwise_and(g, g, mask=binarized_dot)
            masked_b = cv2.bitwise_and(b, b, mask=binarized_dot)
            selected_pixels = np.where(binarized_dot == 255)

            difference_threshold = 0.8

            selected_reds = masked_r[selected_pixels]
            average_red = np.mean(selected_reds)
            max_difference = 0

            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                red_difference = abs(masked_r[y, x] - average_red)
                if red_difference > max_difference:
                    max_difference = red_difference

            cumulative_red = 0
            count = 0

            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                red_difference = abs(masked_r[y, x] - average_red)
                if red_difference < difference_threshold * max_difference:
                    cumulative_red += masked_r[y, x]
                    count += 1

            average_red = cumulative_red / count

            selected_greens = masked_g[selected_pixels]
            average_green = np.mean(selected_greens)
            max_difference = 0

            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                green_difference = abs(masked_g[y, x] - average_green)
                if green_difference > max_difference:
                    max_difference = green_difference

            cumulative_green = 0
            count = 0

            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                green_difference = abs(masked_g[y, x] - average_green)
                if green_difference < difference_threshold * max_difference:
                    cumulative_green += masked_g[y, x]
                    count += 1

            average_green = cumulative_green / count

            selected_blues = masked_b[selected_pixels]
            average_blue = np.mean(selected_blues)
            max_difference = 0

            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                blue_difference = abs(masked_b[y, x] - average_blue)
                if blue_difference > max_difference:
                    max_difference = blue_difference

            cumulative_blue = 0
            count = 0

            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                blue_difference = abs(masked_b[y, x] - average_blue)
                if blue_difference < difference_threshold * max_difference:
                    cumulative_blue += masked_b[y, x]
                    count += 1

            average_blue = cumulative_blue / count

            centroid_x = info[4]
            centroid_y = info[5]
            distances = []
            
            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                centroid_distance = np.sqrt((centroid_x - x) ** 2 + (centroid_y - y) ** 2)
                distances.append(centroid_distance)

            max_distance = max(distances)

            



            red_differences = []
            green_differences = []
            blue_differences = []

            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                red_difference = abs(masked_r[y, x] - average_red)
                red_differences.append(red_difference)
                green_difference = abs(masked_g[y, x] - average_green)
                green_differences.append(green_difference)
                blue_difference = abs(masked_b[y, x] - average_blue)
                blue_differences.append(blue_difference)

            max_red_difference = max(red_differences)
            max_green_difference = max(green_differences)
            max_blue_difference = max(blue_differences)

            weighted_sum_x = 0.0
            weighted_sum_y = 0.0
            total_weight = 0.0

            

            for pixel_idx in range(len(selected_pixels[0])):
                x = selected_pixels[1][pixel_idx]
                y = selected_pixels[0][pixel_idx]
                red_difference = abs(masked_r[y, x] - average_red)
                weight_red = (max_red_difference - red_difference) / max_red_difference
                green_difference = abs(masked_g[y, x] - average_green)
                weight_green = (max_green_difference - green_difference) / max_green_difference
                blue_difference = abs(masked_b[y, x] - average_blue)
                weight_blue = (max_blue_difference - blue_difference) / max_blue_difference

                centroid_distance = np.sqrt((centroid_x - x) ** 2 + (centroid_y - y) ** 2)
                distance_weight = 1.0 - (centroid_distance / max_distance)

                weighted_sum_x += x * weight_red * weight_green * weight_blue * distance_weight
                weighted_sum_y += y * weight_red * weight_green * weight_blue * distance_weight
                total_weight += weight_red * weight_green * weight_blue * distance_weight

            new_centroid_x = weighted_sum_x / total_weight
            new_centroid_y = weighted_sum_y / total_weight
            info[4] = new_centroid_x
            info[5] = new_centroid_y

            updated_info.append(info)

        return updated_info



    
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

        cropped_dots, cropped_dots_with_info = self.extract_dots(ordered_HexaTargets)
        original_labelled_dots = self.label_dots(cropped_dots, ordered_HexaTargets)
        
        updated_info = self.resegment_and_dilate(cropped_dots_with_info)
        updated_labelled_dots = self.label_dots(cropped_dots, None, updated_info)


    

        # labelled_image = self.uniquely_label_dots(uniquely_identified_HexaTargets)

        # return cv2.cvtColor(labelled_image, cv2.COLOR_RGB2BGR)

        return original_labelled_dots, updated_labelled_dots