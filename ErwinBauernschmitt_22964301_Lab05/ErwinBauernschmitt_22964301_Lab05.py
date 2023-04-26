""" CITS4402 COMPUTER VISION - LAB05 WK06
    COLOUR BASED IMAGE SEGMENTATION
    ERWIN BAUERNSCHMITT, 22964301, 05/04/23
"""

### RUNNING NOTES #### 

# The GUI is built from the gui.ui file, which must be in the same directory
# as the Python script. 

# The GUI's window icon should also be in the same directory as it is 
# referenced by the gui.ui file. 

# The script uses the numpy, PyQt6 and cv2 packages. 
# These should be installed for proper running of the script.



### GENERAL COMMENTS ###

# The script performs colour based image segmentation over a range of hues.
# It shows the original image on the left and the segmented one on the right.

# It converts the selected image to a HSV colour space and applies a mask to
# binarise the image into pixels that do fall within the range of hues (white) 
# and pixels that don't (black). Pixels within the hue range are binarised to 
# a white pixel regardless of their saturation or intensity. Tuning these 
# parameters may result in better segmentation for specific images.

# There are two sliders to adjust the upper and lower thresholds of the hue 
# range, and the chosen values are shown at the end of each slider. The 
# segmented image will update automatically as they are changed. The minimum 
# hue range for segmentation is set to 10 by the script.



### PEPPERS TUNED PARAMETERS ###

# To segment (make white) the green peppers:
    # Lower Hue Threshold: 48
    # Upper Hue Threshold: 160



### IRIS-2 TUNED PARAMETERS ###

# To segment (make white) the iris and pupil:
    # Lower Hue Threshold: 25
    # Upper Hue Threshold: 180

# To segment (make white) everything but the iris and the pupil:
    # Lower Hue Threshold: 140
    # Upper Hue Threshold: 359



### IMPORTS ###

import os
import sys
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QLabel, QApplication, QMainWindow
from PyQt6.QtCore import Qt 
from PyQt6.QtGui import QPixmap, QImage, QColor
import cv2



### SCRIPT ###

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, uipath):
        self.uipath = uipath
        self.imageselected = 0      # This is a flag to check if image has been loaded yet.
        self.image_processed = 0    # This is a flag to check if the image has been processed yet.
        self.lower_thresh = 0          # This initialises and sets the lower hue threshold.
        self.upper_thresh = 10         # This initialises and sets the upper hue threshold.
        super(MainWindow, self).__init__()
        uic.loadUi(uipath, self)

        # Links button clicks and slider movements to methods.
        self.load_image_button.clicked.connect(self.loadImage)
        self.lower_thresh_slider.valueChanged.connect(self.updateLowerThresh)
        self.upper_thresh_slider.valueChanged.connect(self.updateUpperThresh)

        # Sets the range of the lower hue threshold slider to be from 0 to 180.
        self.lower_thresh_slider.setRange(0, 359)        
        # Sets the range of the upper hue threshold slider to be from 0 to 180.
        self.upper_thresh_slider.setRange(0, 359)



    def loadImage(self):
        ''' Opens file explorer to allow image selection,
            Reads image in colour,
            Resizes image while keeping aspect ratio,
            Presents image in left panel.
        '''
        # Opens image selection window and extracts file path.
        selected_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", os.path.dirname(os.path.realpath(__file__)), "Image File (*.jpeg *.jpg *.jpe *.jp2 *.png)")
        file_path = selected_file[0]

        # Checks if an image was actually selected.
        if file_path != "":

            # Reads image in colour.
            self.image = cv2.imread(file_path, 1)

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

            # Flags that image has been loaded.
            self.imageselected = 1
            
            # Flags that image has not been processed.
            self.image_processed = 0

            # Runs the initial colour segmentation.
            self.colourSegmentation()

            self.updateLowerThresh()
            self.updateUpperThresh()


    def colourSegmentation(self): 
        """ Segments image based on input hue range,
            Displays segmented binarised image in right panel.
        """ 
        
        lower_thresh_cv2 = int(self.lower_thresh / 2)
        upper_thresh_cv2 = int(self.upper_thresh / 2)

        lower_hsv = np.array([lower_thresh_cv2, 0, 0])
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


    def updateLowerThresh(self):
        """ Reads updated lower threshold value,
            Adjusts sliders to keep values within restrictions,
            Adjusts hue display boxes,
            Triggers a recalculation of colour segmentation.
        """
        # If no image has been processed:
        if self.image_processed == 0: 
            # Then the threshold slider is reset to zero.
            self.lower_thresh_slider.setValue(self.lower_thresh)

        # If an image has been processed:
        else:
            # Read the Lower Hue Threshold slider value into a variable.
            self.lower_thresh = self.lower_thresh_slider.value()

            # If the lower thresh slider is above 170:
            if self.lower_thresh > 349:
                # Then set the slider and it's variable in memory to 170.
                self.lower_thresh_slider.setValue(349)
                self.lower_thresh = 349
            
            # If the lower thresh is too high compared to the upper thresh:
            if self.lower_thresh > (self.upper_thresh - 10): 
                # Set the upper thresh to lower thresh + 10 and update its variable in memory.
                self.upper_thresh_slider.setValue(self.lower_thresh + 10)
                self.upper_thresh = self.lower_thresh + 10

            # Updates the hue labels.
            self.upper_hue_label.setText(f"Hue: {self.upper_thresh}")
            self.lower_hue_label.setText(f"Hue: {self.lower_thresh}")

            # Updates the colour demo boxes.
            upper_color1 = QColor.fromHsv(self.upper_thresh, 255, 0)
            upper_r1, upper_g1, upper_b1, = upper_color1.red(), upper_color1.green(), upper_color1.blue(), 
            upper_color2 = QColor.fromHsv(self.upper_thresh, 255, 255)
            upper_r2, upper_g2, upper_b2, = upper_color2.red(), upper_color2.green(), upper_color2.blue(), 
            upper_color3 = QColor.fromHsv(self.upper_thresh, 0, 255)
            upper_r3, upper_g3, upper_b3, = upper_color3.red(), upper_color3.green(), upper_color3.blue(), 

            self.upper_hue_demo.setStyleSheet(f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgb({upper_r1}, {upper_g1}, {upper_b1}),
                                            stop:0.5 rgb({upper_r2}, {upper_g2}, {upper_b2}),
                                            stop:1 rgba({upper_r3}, {upper_g3}, {upper_b3}, 0));
            """)

            lower_color1 = QColor.fromHsv(self.lower_thresh, 255, 0)
            lower_r1, lower_g1, lower_b1, = lower_color1.red(), lower_color1.green(), lower_color1.blue(), 
            lower_color2 = QColor.fromHsv(self.lower_thresh, 255, 255)
            lower_r2, lower_g2, lower_b2, = lower_color2.red(), lower_color2.green(), lower_color2.blue(), 
            lower_color3 = QColor.fromHsv(self.lower_thresh, 0, 255)
            lower_r3, lower_g3, lower_b3, = lower_color3.red(), lower_color3.green(), lower_color3.blue(), 

            self.lower_hue_demo.setStyleSheet(f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgb({lower_r1}, {lower_g1}, {lower_b1}),
                                            stop:0.5 rgb({lower_r2}, {lower_g2}, {lower_b2}),
                                            stop:1 rgba({lower_r3}, {lower_g3}, {lower_b3}, 0));
            """)

            # Runs colour segmentation again with updated threshold values.
            self.colourSegmentation() 


    def updateUpperThresh(self): 
        """ Reads updated upper threshold value,
            Adjusts sliders to keep values within restrictions,
            Adjusts hue display boxes,
            Triggers a recalculation of colour segmentation.
        """
        # If no image has been processed:
        if self.image_processed == 0: 
            # Then the threshold slider is reset to ten.
            self.upper_thresh_slider.setValue(self.upper_thresh)

        # If an image has been processed:
        else:
            self.upper_thresh = self.upper_thresh_slider.value()
        
            # If the upper thresh slider is below 10:
            if self.upper_thresh < 10:
                # Then set the slider and it's variable in memory to 10.
                self.upper_thresh_slider.setValue(10)
                self.upper_thresh = 10
            
            # If the upper thresh is too low compared to the lower thresh:
            if self.upper_thresh < (self.lower_thresh + 10): 
                # Set the lower thresh to upper thresh - 10 and update its variable in memory.
                self.lower_thresh_slider.setValue(self.upper_thresh - 10)
                self.lower_thresh = self.upper_thresh - 10

            # Updates the hue labels.
            self.upper_hue_label.setText(f"Hue: {self.upper_thresh}")
            self.lower_hue_label.setText(f"Hue: {self.lower_thresh}")

            # Updates the colour demo boxes.
            upper_color1 = QColor.fromHsv(self.upper_thresh, 255, 0)
            upper_r1, upper_g1, upper_b1, = upper_color1.red(), upper_color1.green(), upper_color1.blue(), 
            upper_color2 = QColor.fromHsv(self.upper_thresh, 255, 255)
            upper_r2, upper_g2, upper_b2, = upper_color2.red(), upper_color2.green(), upper_color2.blue(), 
            upper_color3 = QColor.fromHsv(self.upper_thresh, 0, 255)
            upper_r3, upper_g3, upper_b3, = upper_color3.red(), upper_color3.green(), upper_color3.blue(), 

            self.upper_hue_demo.setStyleSheet(f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgb({upper_r1}, {upper_g1}, {upper_b1}),
                                            stop:0.5 rgb({upper_r2}, {upper_g2}, {upper_b2}),
                                            stop:1 rgba({upper_r3}, {upper_g3}, {upper_b3}, 0));
            """)

            lower_color1 = QColor.fromHsv(self.lower_thresh, 255, 0)
            lower_r1, lower_g1, lower_b1, = lower_color1.red(), lower_color1.green(), lower_color1.blue(), 
            lower_color2 = QColor.fromHsv(self.lower_thresh, 255, 255)
            lower_r2, lower_g2, lower_b2, = lower_color2.red(), lower_color2.green(), lower_color2.blue(), 
            lower_color3 = QColor.fromHsv(self.lower_thresh, 0, 255)
            lower_r3, lower_g3, lower_b3, = lower_color3.red(), lower_color3.green(), lower_color3.blue(), 

            self.lower_hue_demo.setStyleSheet(f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgb({lower_r1}, {lower_g1}, {lower_b1}),
                                            stop:0.5 rgb({lower_r2}, {lower_g2}, {lower_b2}),
                                            stop:1 rgba({lower_r3}, {lower_g3}, {lower_b3}, 0));
            """)

            # Runs colour segmentation again with updated threshold values.
            self.colourSegmentation() 


def main():

    # Determines path of gui
    uiname = "gui.ui"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    uipath = dir_path + "\\" + uiname

    # Runs gui
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(uipath)
    main.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

            


