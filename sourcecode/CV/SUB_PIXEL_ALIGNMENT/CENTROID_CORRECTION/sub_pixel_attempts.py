### TWO-STAGE AVERAGE

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
                # Dilate image once with a square 3x3 structuring element and show image with title.
                dilated_red_segments = cv2.dilate(red_segments, kernel, iterations = 1)

                # Creating an HSV copy of the dot
                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2RGB)

                dilated_dot_with_info = (hsv_dot, dilated_red_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

            if colour == "green":
                green_segments1 = self.colour_segment_image("green1", dot)
                green_segments2 = self.colour_segment_image("green2", dot)
                green_segments = cv2.bitwise_or(green_segments1, green_segments2)   

                kernel = np.ones((3, 3))
                # Dilate image once with a square 3x3 structuring element and show image with title.
                dilated_green_segments = cv2.dilate(green_segments, kernel, iterations = 1)

                # Creating an HSV copy of the dot
                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2RGB)
                
                dilated_dot_with_info = (hsv_dot, dilated_green_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

            if colour == "blue":
                blue_segments1 = self.colour_segment_image("blue1", dot)
                blue_segments2 = self.colour_segment_image("blue2", dot)
                blue_segments = cv2.bitwise_or(blue_segments1, blue_segments2)

                kernel = np.ones((3, 3))
                # Dilate image once with a square 3x3 structuring element and show image with title.
                dilated_blue_segments = cv2.dilate(blue_segments, kernel, iterations = 1)

                # Creating an HSV copy of the dot
                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2RGB)
                
                dilated_dot_with_info = (hsv_dot, dilated_blue_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

        updated_info = []

        for (dot, binarized_dot, info) in dilated_dots_with_info:
            # Convert the dot to float32 for precise calculations
            dot = dot.astype(np.float32)

            # Normalize the intensity
            normalized_dot = dot / np.linalg.norm(dot, axis=2, keepdims=True)

            # Split the normalized RGB image into individual channels
            r, g, b = cv2.split(normalized_dot)

            # Apply the mask on each channel
            masked_r = cv2.bitwise_and(r, r, mask=binarized_dot)
            masked_g = cv2.bitwise_and(g, g, mask=binarized_dot)
            masked_b = cv2.bitwise_and(b, b, mask=binarized_dot)

            # Identifies pixels in selection
            selected_pixels = np.where(binarized_dot == 255)

            difference_threshold = 0.8

            # Calculates average red component
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
            


            # Calculates average green component
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




            # Calculates average blue component
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





            # Initialize variables
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

                weight_red = (max_red_difference - abs(masked_r[y, x] - average_red)) / max_red_difference
                weight_green = (max_green_difference - abs(masked_g[y, x] - average_green)) / max_green_difference
                weight_blue = (max_blue_difference - abs(masked_b[y, x] - average_blue)) / max_blue_difference

                # Accumulate the weighted sums
                # if weight_red > 0.5 and weight_green > 0.5 and weight_blue > 0.5:
                weighted_sum_x += x * weight_red * weight_green * weight_blue
                weighted_sum_y += y * weight_red * weight_green * weight_blue
                total_weight += weight_red * weight_green * weight_blue

            centroid_x = weighted_sum_x / total_weight
            centroid_y = weighted_sum_y / total_weight

            info[4] = centroid_x
            info[5] = centroid_y

            updated_info.append(info)


        return updated_info




### TWO STAGE AVERAGE WITH DISTANCE WEIGHTING

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

                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2HSV)
                dilated_dot_with_info = (hsv_dot, dilated_red_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

            if colour == "green":
                green_segments1 = self.colour_segment_image("green1", dot)
                green_segments2 = self.colour_segment_image("green2", dot)
                green_segments = cv2.bitwise_or(green_segments1, green_segments2)

                kernel = np.ones((3, 3))
                dilated_green_segments = cv2.dilate(green_segments, kernel, iterations=1)

                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2HSV)
                dilated_dot_with_info = (hsv_dot, dilated_green_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

            if colour == "blue":
                blue_segments1 = self.colour_segment_image("blue1", dot)
                blue_segments2 = self.colour_segment_image("blue2", dot)
                blue_segments = cv2.bitwise_or(blue_segments1, blue_segments2)

                kernel = np.ones((3, 3))
                dilated_blue_segments = cv2.dilate(blue_segments, kernel, iterations=1)

                hsv_dot = cv2.cvtColor(dot.copy(), cv2.COLOR_BGR2HSV)
                dilated_dot_with_info = (hsv_dot, dilated_blue_segments, info)
                dilated_dots_with_info.append(dilated_dot_with_info)

            updated_info = []

            for hsv_dot, dilated_segment, info in dilated_dots_with_info:
                (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(dilated_segment, connectivity = 8, ltype = cv2.CV_32S)
                (cX, cY) = centroids[1]

                info[4] = cX
                info[5] = cY

                updated_info.append(info)



        return updated_info