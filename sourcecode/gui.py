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
        self.previous_image_button.clicked.connect(self.previous_image)
        self.next_image_button.clicked.connect(self.next_image)
        self.delete_image_button.clicked.connect(self.delete_image)

        self.images = []
        self.labelled_images = []
        self.images_with_info = []
        self.current_image = 0


    def image_exists(self, new_image, image_list):
        """ Compares a given image to a list of images.
            Returns true if an exact match is found, false otherwise.
        """
        for existing_image in image_list:
            if np.array_equal(new_image, existing_image):
                return True
        return False


    def load_image(self):
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
            # Reads image in colour and displays resized image in left panel.
            new_image = cv2.imread(filename = file_path)

            # If the selected image has already been loaded:
            if self.image_exists(new_image, self.images):
                print("This image has already been processed, please choose a different one")
            
            # If the selected image has not already been loaded:
            else:
                # Performs HexaTarget identification.
                identifier_output = HexaTargetIdentifier(new_image)
                labelled_image = identifier_output.labelled_image.copy()
                HexaTargets = identifier_output.HexaTargets.copy()

                # Records the image and the HexaTarget identification outputs.
                self.images.append(new_image)
                self.labelled_images.append(labelled_image)
                self.images_with_info.append([new_image, labelled_image, HexaTargets])
                
                # Displays the images with labelled HexaTargets.
                self.display_image(new_image, self.original_image, "BGR")
                self.display_image(labelled_image, self.processed_image, "RGB")
                self.display_full_image(labelled_image, "RGB", "Labelled HexaTargets")

                # 
                self.current_image = len(self.images)
                self.perspective_number.setText(f"Perspective {self.current_image} of {len(self.images)}")
                self.original_label.setText(f"Perspective {self.current_image}: Original Image")
                self.processed_label.setText(f"Perspective {self.current_image}: Labelled HexaTargets")



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


    def next_image(self):
        if len(self.images) > 1:
            if len(self.images) > self.current_image:
                self.current_image += 1
                original_image = self.images[self.current_image - 1]
                processed_image = self.labelled_images[self.current_image - 1]
                self.display_image(original_image, self.original_image, "BGR")
                self.display_image(processed_image, self.processed_image, "RGB")
                self.perspective_number.setText(f"Perspective {self.current_image} of {len(self.images)}")
                self.original_label.setText(f"Perspective {self.current_image}: Original Image")
                self.processed_label.setText(f"Perspective {self.current_image}: Labelled HexaTargets")


    def previous_image(self):
        if len(self.images) > 1:
            if self.current_image > 1:
                self.current_image -= 1
                original_image = self.images[self.current_image - 1]
                processed_image = self.labelled_images[self.current_image - 1]
                self.display_image(original_image, self.original_image, "BGR")
                self.display_image(processed_image, self.processed_image, "RGB")
                self.perspective_number.setText(f"Perspective {self.current_image} of {len(self.images)}")
                self.original_label.setText(f"Perspective {self.current_image}: Original Image")
                self.processed_label.setText(f"Perspective {self.current_image}: Labelled HexaTargets")


    def delete_image(self):
        # If there are at least two images in the list: 
        if len(self.images) >= 2:
            # If the current image is the last image in the list:
            if self.current_image == len(self.images):
                # Displaus the second-last image in the list as the current image.
                self.current_image -= 1
                original_image = self.images[self.current_image - 1]
                processed_image = self.labelled_images[self.current_image - 1]
                self.display_image(original_image, self.original_image, "BGR")
                self.display_image(processed_image, self.processed_image, "RGB")
                self.perspective_number.setText(f"Perspective {self.current_image} of {len(self.images) - 1}")
                self.original_label.setText(f"Perspective {self.current_image}: Original Image")
                self.processed_label.setText(f"Perspective {self.current_image}: Labelled HexaTargets")

                # Deletes the last image in the list
                del self.images[self.current_image]
                del self.labelled_images[self.current_image]
                del self.images_with_info[self.current_image]

            # If the current image is not the last image in the list:
            elif self.current_image < len(self.images):
                # Removes the current image and its processed outputs from memory.
                del self.images[self.current_image - 1]
                del self.labelled_images[self.current_image - 1]
                del self.images_with_info[self.current_image - 1]

                # Displays the next image in the list as the current image.
                original_image = self.images[self.current_image - 1]
                processed_image = self.labelled_images[self.current_image - 1]
                self.display_image(original_image, self.original_image, "BGR")
                self.display_image(processed_image, self.processed_image, "RGB")
                self.perspective_number.setText(f"Perspective {self.current_image} of {len(self.images)}")
                self.original_label.setText(f"Perspective {self.current_image}: Original Image")
                self.processed_label.setText(f"Perspective {self.current_image}: Labelled HexaTargets")

        # If there is only one image in the list:
        elif len(self.images) == 1:
            # Resets current_image and all labels to zero.
            self.current_image = 0
            self.perspective_number.setText(f"Perspective 0 of 0")
            self.original_label.setText(f"Perspective 0: Original Image")
            self.processed_label.setText(f"Perspective 0: Labelled HexaTargets")

            # Removes the only image and its processed outputs from memory.
            del self.images[0]
            del self.labelled_images[0]
            del self.images_with_info[0]

            # Resets the display image labels.
            self.original_image.clear()
            self.processed_image.clear()







