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
            #file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
            print(file_path)

            # Checks if an image was actually selected.
            if file_path != "":

                # Reads image in colour.
                self.image = cv2.imread(filename = file_path)

                # Resizes image to display window size, keeping aspect ratio.
                if (self.image.shape[1] / 371) > (self.image.shape[0] / 271):
                    width = 371
                    height = int(self.image.shape[0] / self.image.shape[1] * 371)
                    dim = (width, height)
                else:
                    height = 271
                    width = int(self.image.shape[1] / self.image.shape[0] * 271)
                    dim = (width, height)
                
                # Converts cv2 image to QImage for display.
                self.image = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)
                self.colourimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

                h, w, ch = self.colourimage.shape
                qimage = QImage(self.colourimage.data, w, h, ch * w, QImage.Format.Format_RGB888)
        
                # Displays original colour image in left panel.
                self.pixmap = QPixmap.fromImage(qimage)
                self.original_image.setPixmap(self.pixmap)

                # Converts cv2 image from the BGR to the HSV colour space for further processing.
                self.hsvimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

                blue = self.colourSegmentation("blue")
                red = self.colourSegmentation("red")
                green = self.colourSegmentation("green")
                combined = blue + red + green

                # Converts cv2 image to QImage for display.
                im_np = np.array(combined)  
                qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0], combined.strides[0], QImage.Format.Format_Grayscale8)

                # Displays processed image in the right panel.
                self.pixmap = QPixmap.fromImage(qimage)
                self.processed_image.setPixmap(self.pixmap)

                blue_boxes = self.object_analysis(blue)
                red_boxes = self.object_analysis(red)
                green_boxes = self.object_analysis(green)
                for i in range(len(red_boxes)):
                    boxes = cv2.rectangle(self.image, (red_boxes[i][0],red_boxes[i][1]), (red_boxes[i][0] + red_boxes[i][2], red_boxes[i][1] - red_boxes[i][3]), [255, 0,0])
                for i in range(len(blue_boxes)):
                    boxes = cv2.rectangle(self.image, (blue_boxes[i][0],blue_boxes[i][1]), (blue_boxes[i][0] + blue_boxes[i][2], blue_boxes[i][1] - blue_boxes[i][3]), [0, 0,255])
                for i in range(len(green_boxes)):
                    boxes = cv2.rectangle(self.image, (green_boxes[i][0],green_boxes[i][1]), (green_boxes[i][0] + green_boxes[i][2], green_boxes[i][1] - green_boxes[i][3]), [0, 255,0])

                im_np = np.array(boxes)  
                qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0], boxes.strides[0], QImage.Format.Format_RGB888)

                # Displays processed image in the right panel.
                self.pixmap = QPixmap.fromImage(qimage)
                self.processed_image.setPixmap(self.pixmap)


    def colourSegmentation(self, request): 
            """ Segments image based on input hue range,
                Displays segmented binarised image in right panel.
            """ 

            colours = {"blue": [90,130,100,60], "red": [169,179,100,60], "green": [45,85,90,50] }

            lower_thresh_cv2 = colours[request][0]
            upper_thresh_cv2 = colours[request][1]

            lower_hsv = np.array([lower_thresh_cv2, colours[request][2], colours[request][3]])
            upper_hsv = np.array([upper_thresh_cv2, 255, 255])


            thresholding_mask = cv2.inRange(self.hsvimage, lower_hsv, upper_hsv)

            segmented_image = cv2.bitwise_and(self.hsvimage, self.hsvimage, mask = thresholding_mask)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
            ret, segmented_image = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY)

            # Converts cv2 image to QImage for display.
            im_np = np.array(segmented_image)  
            qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0], segmented_image.strides[0], QImage.Format.Format_Grayscale8)

            # Displays processed image in the right panel.
            self.pixmap = QPixmap.fromImage(qimage)
            self.processed_image.setPixmap(self.pixmap)

            # Sets flag to indicate that image has been processed.
            self.image_processed = 1
            return(segmented_image)


    def object_analysis(self, image):
        output = cv2.connectedComponentsWithStats(image, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        viable = []
        for i in range(1, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(
                    i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format( i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            if 4 <= area <= 20: 
                viable.append([x,y,w,h])

        return(viable)

if __name__ == '__main__':
    uiname = "gui.ui"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    uipath = os.path.join(dir_path, uiname)

    # Runs gui
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(uipath)
    main.show()
    sys.exit(app.exec())