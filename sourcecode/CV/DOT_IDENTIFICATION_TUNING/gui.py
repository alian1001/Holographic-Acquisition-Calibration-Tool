import os
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QPixmap, QImage
import cv2
from functions import ImageSegmenter


class CalibratorGUI(QtWidgets.QWidget):
    """ Defines all of the Automatic Calibrator GUI functionality.
    """
    def __init__(self, uipath):
        super().__init__()
        uic.loadUi(uipath, self)
        self.load_image_button.clicked.connect(self.load_images)
        self.previous_image_button.clicked.connect(self.previous_image)
        self.next_image_button.clicked.connect(self.next_image)

                            # lower       upper
        self.colours = {"blue1":  [[ 98, 29,  0],[141,255,113]], 
                        "blue2":  [[ 98, 97, 96],[130,255,255]],
                        "red1":   [[150, 35,  0],[179,255,150]], 
                        "red2":   [[150,100,100],[179,255,223]],
                        "red3":   [[  0, 35, 60],[ 10,255,223]],
                        "green1": [[ 37,  0,  0],[102,160,100]],
                        "green2": [[ 37, 42,101],[ 91,255,168]]}

        self.images = []
        self.segmented_images = []
        self.current_image = 0



    def load_images(self):
        ''' Opens file explorer to allow multiple image selection,
            Reads each image in colour,
            Resizes each image while keeping aspect ratio,
            Presents each image in left panel.
        '''
        # Opens image selection window and extracts file paths.
        selected_files = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Images", os.path.dirname(os.path.realpath(__file__)), "Image File (*.jpeg *.jpg *.jpe *.jp2 *.png)")
        file_paths = selected_files[0]

        # Checks if any images were actually selected.
        if file_paths:
            for file_path in file_paths:
                # Reads image in colour and displays resized image in left panel.
                new_image = cv2.imread(filename = file_path)

                # Performs image segmentation.
                output = ImageSegmenter(new_image, self.colours)
                segmented_image = output.segmented_image.copy()

                # Records the original image and the segmented image.
                self.images.append(new_image)
                self.segmented_images.append(segmented_image)
            
            # Displays the images.
            self.display_image(self.images[0], self.original_image, "BGR")
            self.display_image(self.segmented_images[0], self.processed_image, "BGR")

            # Updates the labels.
            self.current_image = 1
            self.perspective_number.setText(f"HexaTarget {self.current_image} of {len(self.images)}")
            self.original_label.setText(f"HexaTarget {self.current_image}: Original Image")
            self.processed_label.setText(f"HexaTarget {self.current_image}: Segmented Image")

            self.display_full_image(segmented_image, "BGR", "Dots")




    def display_image(self, image, location, format):
        """ Displays the given image at the given location. 
            (format = "RGB", "HSV", "BGR", or "grey")
        """
        # Deep copies image to avoid modifying the original image.
        resized_image = image.copy()

        # Resizes image to display window size, keeping aspect ratio.
        if (resized_image.shape[1] / 560) > (resized_image.shape[0] / 400):
            width = 560
            height = int(resized_image.shape[0] / resized_image.shape[1] * 560)
            dimensions = (width, height)
        else:
            height = 400
            width = int(resized_image.shape[1] / resized_image.shape[0] * 400)
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


    def next_image(self):
        if len(self.images) > 1:
            if len(self.images) > self.current_image:
                self.current_image += 1

                # Retrieves original version of currently displayed image.
                original_image = self.images[self.current_image - 1]

                # Performs image segmentation.
                output = ImageSegmenter(original_image, self.colours)
                segmented_image = output.segmented_image.copy()

                # Records the newly segmented image.
                self.segmented_images.append(segmented_image)

                # Displays the images.
                self.display_image(original_image, self.original_image, "BGR")
                self.display_image(segmented_image, self.processed_image, "BGR")

                # Updates the labels.
                self.perspective_number.setText(f"HexaTarget {self.current_image} of {len(self.images)}")
                self.original_label.setText(f"HexaTarget {self.current_image}: Original Image")
                self.processed_label.setText(f"HexaTarget {self.current_image}: Segmented Image")

                self.display_full_image(segmented_image, "BGR", "Dots")


    def previous_image(self):
        if len(self.images) > 1:
            if self.current_image > 1:
                self.current_image -= 1

                # Retrieves original version of currently displayed image.
                original_image = self.images[self.current_image - 1]

                # Performs image segmentation.
                output = ImageSegmenter(original_image, self.colours)
                segmented_image = output.segmented_image.copy()

                # Records the newly segmented image.
                self.segmented_images.append(segmented_image)
                
                # Displays the images.
                self.display_image(original_image, self.original_image, "BGR")
                self.display_image(segmented_image, self.processed_image, "BGR")

                # Updates the labels.
                self.perspective_number.setText(f"HexaTarget {self.current_image} of {len(self.images)}")
                self.original_label.setText(f"HexaTarget {self.current_image}: Original Image")
                self.processed_label.setText(f"HexaTarget {self.current_image}: Segmented Image")

                self.display_full_image(segmented_image, "BGR", "Dots")


    # def update_segmentation(self):
    #     # Retrieves original version of currently displayed image.
    #     original_image = self.images[self.current_image - 1]

    #     # Performs image segmentation.
    #     output = ImageSegmenter(original_image, self.colours)
    #     segmented_image = output.segmented_image.copy()

    #     # Records the newly segmented image.
    #     self.segmented_images.append(segmented_image)
        
    #     # Displays the newly segmented image.
    #     self.display_image(segmented_image, self.processed_image, "BGR")

    #     # Print the new colours dictionary. 
    #     print(self.colours)




