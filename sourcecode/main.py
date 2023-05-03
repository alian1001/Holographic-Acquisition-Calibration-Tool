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
        self.colour_segment_button.clicked.connect(self.colourSegmentation)

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

                self.colourSegmentation()


    def colourSegmentation(self): 
            """ Segments image based on input hue range,
                Displays segmented binarised image in right panel.
            """ 
            
            #lower_thresh_cv2 = int(self.lower_thresh / 2)
            #upper_thresh_cv2 = int(self.upper_thresh / 2)


            # BLUE VALUES

            lower_thresh_cv2 = 90
            upper_thresh_cv2 = 130

            lower_hsv = np.array([lower_thresh_cv2, 100, 60])
            upper_hsv = np.array([upper_thresh_cv2, 255, 255])

            # RED VALUES

            # lower_thresh_cv2 = 169
            # upper_thresh_cv2 = 179

            # lower_hsv = np.array([lower_thresh_cv2, 100, 60])
            # upper_hsv = np.array([upper_thresh_cv2, 255, 255])

            # GREEN VALUES 

            # lower_thresh_cv2 = 45
            # upper_thresh_cv2 = 85

            # lower_hsv = np.array([lower_thresh_cv2, 90, 50])
            # upper_hsv = np.array([upper_thresh_cv2, 255, 255])


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

if __name__ == '__main__':
    uiname = "gui.ui"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    uipath = os.path.join(dir_path, uiname)

    # Runs gui
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(uipath)
    main.show()
    sys.exit(app.exec())