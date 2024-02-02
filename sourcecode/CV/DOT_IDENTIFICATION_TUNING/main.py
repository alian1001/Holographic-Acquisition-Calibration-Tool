"""
    HOLOGRAPHIC ACQUISITION RIG AUTOMATIC CALIBRATION
    ERWIN BAUERNSCHMITT, ALIAN HAIDAR, LUKE KIRKBY
"""

### RUNNING NOTES ###



### GENERAL COMMENTS ### 



import sys
import os
import numpy as np
from PyQt6.QtWidgets import QApplication
from gui import CalibratorGUI


def main():
    """ Launches the Automatic Callibrator GUI.
    """
    # Identifies gui.ui path
    uiname = "gui.ui"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    uipath = os.path.join(dir_path, uiname)

    # Runs the GUI
    app = QApplication(sys.argv)
    main = CalibratorGUI(uipath)
    main.show()
    sys.exit(app.exec())    


if __name__ == '__main__':
    main()