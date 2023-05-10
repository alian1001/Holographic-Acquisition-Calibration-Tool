#Main file for CITS4402 Project

import sys
import random
# from PySide6 import QtCore, QtWidgets, QtGui
import os
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QLabel, QApplication, QMainWindow
from PyQt6.QtCore import Qt 
from PyQt6.QtGui import QPixmap, QImage, QColor
import cv2
import skimage
# import functions
# from functions import loadImage

class MainWindow(QtWidgets.QWidget):
    def __init__(self, uipath):
        super().__init__()
        uic.loadUi(uipath, self)
        
        self.load_image_button.clicked.connect(self.loadImage)

    def loadImage(self):
            ''' Opens file explorer to allow image selection,
                Reads image in colour,
                Resizes image while keeping aspect ratio,
                Presents image in left panel.
            '''
            # Opens image selection window and extracts file path.
            selected_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", os.path.dirname(os.path.realpath(__file__)), "Image File (*.jpeg *.jpg *.jpe *.jp2 *.png)")
            file_path = selected_file[0]
            print(file_path)

            # Checks if an image was actually selected.
            if file_path != "":

                # Reads image in colour.
                self.image = cv2.imread(filename = file_path)
                self.resized_image = self.image.copy()

                # Resizes image to display window size, keeping aspect ratio.
                if (self.resized_image.shape[1] / 371) > (self.resized_image.shape[0] / 271):
                    width = 371
                    height = int(self.resized_image.shape[0] / self.resized_image.shape[1] * 371)
                    dim = (width, height)
                else:
                    height = 271
                    width = int(self.resized_image.shape[1] / self.resized_image.shape[0] * 271)
                    dim = (width, height)
                
                # Converts cv2 image to QImage for display.
                self.resized_image = cv2.resize(self.resized_image, dim, interpolation = cv2.INTER_AREA)
                self.colourimage = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2RGB)

                self.display_colour(self.colourimage,self.original_image)

                # Converts cv2 image from the BGR to the HSV colour space for further processing.
                self.hsvimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

                blue = self.colourSegmentation("blue")
                red = self.colourSegmentation("red")
                green = self.colourSegmentation("green")
                combined = blue + red + green

                self.display_greyscale(combined, self.processed_image)

                blue_boxes = self.object_analysis(blue)
                red_boxes = self.object_analysis(red)
                green_boxes = self.object_analysis(green)


                clusters = self.groupings([blue_boxes, green_boxes, red_boxes])

                sorted_clusters = self.cluster_analysis(clusters)
                cluster_names = self.name_clusters(sorted_clusters)
                print(cluster_names)
                for i in range(len(sorted_clusters)):
                    for j in range(len(sorted_clusters[i])):

                        clust = cv2.rectangle(self.image, (sorted_clusters[i][j][0],sorted_clusters[i][j][1]), (sorted_clusters[i][j][0] + sorted_clusters[i][j][2], sorted_clusters[i][j][1] + sorted_clusters[i][j][3]), sorted_clusters[i][j][-1])
                        #self.display_colour(clust, self.render_image)
                    named = cv2.putText(self.image, cluster_names[i], org= [sorted_clusters[i][0][0],sorted_clusters[i][0][1]] ,fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.5, color=[255,0,0])
                    self.display_colour(named, self.render_image)
                







    def colourSegmentation(self, request): 
            """ Segments image based on input hue range,
                Displays segmented binarised image in right panel.
            """ 

            colours = {"blue": [90,130,100,60], "red": [169,179,100,60], "green": [40,90,45,40] }

            lower_thresh_cv2 = colours[request][0]
            upper_thresh_cv2 = colours[request][1]

            lower_hsv = np.array([lower_thresh_cv2, colours[request][2], colours[request][3]])
            upper_hsv = np.array([upper_thresh_cv2, 255, 255])


            thresholding_mask = cv2.inRange(self.hsvimage, lower_hsv, upper_hsv)

            segmented_image = cv2.bitwise_and(self.hsvimage, self.hsvimage, mask = thresholding_mask)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
            ret, segmented_image = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY)

            self.display_greyscale(segmented_image, self.processed_image)

            # Sets flag to indicate that image has been processed.
            self.image_processed = 1
            return(segmented_image)


    def object_analysis(self, image):
        output = cv2.connectedComponentsWithStats(image, cv2.CV_32S)

        (numLabels, labels, stats, centroids) = output

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        viable = []

        for i in range(1, numLabels): #starts at 1 to exlude the background
            # Extract the connected component statistics and centroid for the current label
            # Centroid info currently unused
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            if 12 <= area <= 300: 
                viable.append([x,y,w,h,cX,cY])

        return(viable)

    def display_colour(self, image, location):
        large_display_image = image.copy()
        large_display_image = cv2.cvtColor(large_display_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Full Resolution", large_display_image)
        
        resized_image = image.copy()

        # Resizes image to display window size, keeping aspect ratio.
        if (resized_image.shape[1] / 371) > (resized_image.shape[0] / 271):
            width = 371
            height = int(resized_image.shape[0] / resized_image.shape[1] * 371)
            dim = (width, height)
        else:
            height = 271
            width = int(resized_image.shape[1] / resized_image.shape[0] * 271)
            dim = (width, height)
        
        # Converts cv2 image to QImage for display.
        resized_image = cv2.resize(resized_image, dim, interpolation = cv2.INTER_AREA)
        im_np = np.array(resized_image)  
        qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0], resized_image.strides[0], QImage.Format.Format_RGB888)

        # Displays processed image in the right panel.
        self.pixmap = QPixmap.fromImage(qimage)
        location.setPixmap(self.pixmap)


    def display_greyscale(self, image, location):
        resized_image = image.copy()

        # Resizes image to display window size, keeping aspect ratio.
        if (resized_image.shape[1] / 371) > (resized_image.shape[0] / 271):
            width = 371
            height = int(resized_image.shape[0] / resized_image.shape[1] * 371)
            dim = (width, height)
        else:
            height = 271
            width = int(resized_image.shape[1] / resized_image.shape[0] * 271)
            dim = (width, height)
        
        # Converts cv2 image to QImage for display.
        resized_image = cv2.resize(resized_image, dim, interpolation = cv2.INTER_AREA)
        im_np = np.array(resized_image)  
        qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0], resized_image.strides[0], QImage.Format.Format_Grayscale8)

        # Displays processed image in the right panel.
        self.pixmap = QPixmap.fromImage(qimage)
        location.setPixmap(self.pixmap)


    def groupings(self, object_list):
        blue = object_list[0]
        green = object_list[1]
        red = object_list[2]
        clusters = []

        for i in range(len(blue)):
            blue[i].append([0, 0, 255])
            cluster = [blue[i]]


            for j in range(len(green)):

                if np.absolute(green[j][4] - blue[i][4]) <= 65:
                    if np.absolute(green[j][5] - blue[i][5]) <= 65 :
                        green[j].append([0, 255,0])
                        cluster.append(green[j])

            for k in range(len(red)):
                if np.absolute(red[k][4] - blue[i][4]) <= 65:
                    if np.absolute(red[k][5] - blue[i][5]) <= 65 :
                        red[k].append([255, 0,0])
                        cluster.append(red[k])

            if(len(cluster) == 6):
                clusters.append(cluster)

        return clusters

    def cluster_analysis(self, clusters):
        ordered_clusters = []
        

        for cluster in clusters:
            cluster = cluster.copy()

            blue = cluster[0]
            blue_x_centroid = blue[4]
            blue_y_centroid = blue[5]
            cluster.pop(0)

            temp_clust = [blue,0,0,0,0,0]
            # temp_clust = [point_0, point_1, point_2, point_3, point_4, point_5] (clockwise from blue).

            point_id_sequence = [3, 1, 5, 2, 4]

            for point_number in point_id_sequence:

                if point_number == 1:
                    min_y = blue_y_centroid + 1000
                    confirmed_point = 0
                    for i in range(len(cluster)):
                        if cluster[i][4] > blue_x_centroid:
                            if cluster[i][5] < min_y:
                                min_y = cluster[i][5]
                                temp_clust[1] = cluster[i]
                                confirmed_point = i
                    cluster.pop(confirmed_point)


                if point_number == 2:
                    for i in range(len(cluster)):
                        if cluster[i][4] > blue_x_centroid:
                            temp_clust[2] = cluster[i]

                if point_number == 3:
                    max_y = blue_y_centroid
                    confirmed_point = 0
                    for i in range(len(cluster)):
                        if cluster[i][5] > max_y:
                            max_y = cluster[i][5]
                            temp_clust[3] = cluster[i]
                            confirmed_point = i
                    cluster.pop(confirmed_point)
                    
                    

                if point_number == 4:
                    for i in range(len(cluster)):
                        if cluster[i][4] < blue_x_centroid:
                            temp_clust[4] = cluster[i]

                if point_number == 5:
                    min_y = blue_y_centroid + 1000
                    confirmed_point = 0
                    for i in range(len(cluster)):
                        if cluster[i][4] < blue_x_centroid:
                            if cluster[i][5] < min_y:
                                min_y = cluster[i][5]
                                temp_clust[5] = cluster[i]
                                confirmed_point = i
                    cluster.pop(confirmed_point)

            ordered_clusters.append(temp_clust)
        return ordered_clusters
                    

                
    def name_clusters(self, sorted_clusters):
        names = []
        for cluster in sorted_clusters:
            name = "HexaTarget_"
            for point in cluster:
                point_id = point[-1]
                if point_id == [255,0,0]:
                    colour = "R"
                if point_id == [0,255,0]:
                    colour = "G"
                if point_id == [0,0,255]:
                    colour = "B"
                name += colour
            names.append(name)
        return names




if __name__ == '__main__':
    uiname = "gui.ui"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    uipath = os.path.join(dir_path, uiname)

    # Runs gui
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(uipath)
    main.show()
    sys.exit(app.exec())