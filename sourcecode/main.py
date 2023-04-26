#Main file for CITS4402 Project

import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui
import os
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QLabel, QApplication, QMainWindow
from PyQt6.QtCore import Qt 
from PyQt6.QtGui import QPixmap, QImage, QColor
import cv2
import functions
from functions import loadImage

class MainWindow(QtWidgets.QWidget):
    def __init__(self, uipath):
        super().__init__()
        uic.loadUi(uipath, self)

        self.load_image_button.clicked.connect(self.loadImage)

    #     self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир", "Hello, bro"]

    #     self.button = QtWidgets.QPushButton("Click me!")
    #     self.text = QtWidgets.QLabel("Hello World",
    #                                  alignment=QtCore.Qt.AlignCenter)

    #     self.layout = QtWidgets.QVBoxLayout(self)
    #     self.layout.addWidget(self.text)
    #     self.layout.addWidget(self.button)

    #     self.button.clicked.connect(self.magic)

    # @QtCore.Slot()
    # def magic(self):
    #     self.text.setText(random.choice(self.hello))

# if __name__ == "__main__":
#     app = QtWidgets.QApplication([])

#     widget = MyWidget()
#     widget.resize(800, 600)
#     widget.show()

#     sys.exit(app.exec())
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


if __name__ == '__main__':
    uiname = "gui.ui"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    uipath = dir_path + "\\" + uiname

    # Runs gui
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(uipath)
    main.show()
    sys.exit(app.exec())