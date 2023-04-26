import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
from scipy import ndimage
from skimage import img_as_ubyte, img_as_int
import os
import sys
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QLabel, QApplication, QMainWindow
from PyQt6.QtCore import Qt 
from PyQt6.QtGui import QPixmap, QImage, QColor
import cv2


def load_image(self):
     # Open a file selection dialog box to choose an image file
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        
        # Load the chosen image using PIL
        self.original_image = Image.open(file_path)
        
        # Resize the image to fit in the label
        width, height = self.original_image.size
        max_size = 250
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        self.original_image = self.original_image.resize((new_width, new_height))
        
        # Convert the image to Tkinter format and display it on the left side
        photo = ImageTk.PhotoImage(self.original_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

def loadImage(self):
        ''' Opens file explorer to allow image selection,
            Reads image in colour,
            Resizes image while keeping aspect ratio,
            Presents image in left panel.
        '''
        # Opens image selection window and extracts file path.
        #selected_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", os.path.dirname(os.path.realpath(__file__)), "Image File (*.jpeg *.jpg *.jpe *.jp2 *.png)")
        #file_path = selected_file[0]
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
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

