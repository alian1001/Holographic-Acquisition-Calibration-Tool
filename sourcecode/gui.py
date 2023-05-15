import os
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QPixmap, QImage
import cv2
from functions import HexaTargetIdentifier


class CalibratorGUI(QtWidgets.QWidget):
    """ Defines all of the Automatic Calibrator GUI functionality.
    """
    def __init__(self, uipath):
        super().__init__()
        uic.loadUi(uipath, self)
        self.load_image_button.clicked.connect(self.load_image)


    def load_image(self):
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
            # Reads image in colour and displays resized image in left panel.
            self.image = cv2.imread(filename = file_path)
            self.display_image(self.image, self.original_image, "BGR")

            # Converts cv2 image from the BGR to the HSV colour space for further processing.
            self.hsvimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

            # Performs HexaTarget identification.
            image_1 = HexaTargetIdentifier(self.image)
            
            # Displays image with labelled HexaTargets in middle panel
            self.display_image(image_1.labelled_image, self.processed_image, "RGB")
            self.display_full_image(image_1.labelled_image, "RGB", "Labelled HexaTargets")

            # Prints each HexaTarget
            for HexaTarget in image_1.HexaTargets:
                print(HexaTarget)


    def display_image(self, image, location, format):
        """ Displays the given image at the given location. 
            (format = "RGB", "HSV", "BGR", or "grey")
        """
        # Deep copies image to avoid modifying the original image.
        resized_image = image.copy()

        # Resizes image to display window size, keeping aspect ratio.
        if (resized_image.shape[1] / 371) > (resized_image.shape[0] / 271):
            width = 371
            height = int(resized_image.shape[0] / resized_image.shape[1] * 371)
            dimensions = (width, height)
        else:
            height = 271
            width = int(resized_image.shape[1] / resized_image.shape[0] * 271)
            dimensions = (width, height)

        # Converts colour image's format to QImage standard (RGB)
        if format == "BGR":
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        if format == "HSV":
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_HSV2BGR)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Converts cv2 image to QImage for display.
        resized_image = cv2.resize(resized_image, dimensions, interpolation = cv2.INTER_AREA)
        im_np = np.array(resized_image) 

        if format == "grey":
            qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0], resized_image.strides[0], QImage.Format.Format_Grayscale8)
        else: 
            qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0], resized_image.strides[0], QImage.Format.Format_RGB888)

        # Displays processed image in the specified panel.
        self.pixmap = QPixmap.fromImage(qimage)
        location.setPixmap(self.pixmap)


    def display_full_image(self, image, format, window_name):
        """ Displays the given image at full size in a separate window.
            (format = "RGB", "HSV", "BGR", or "grey")
        """
        # Deep copies image to avoid modifying the original image.
        display_image = image.copy()

        if format == "RGB":
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        cv2.imshow(window_name, display_image)