# CITS4402 Computer Vision
## ERWIN BAUERNSCHMITT, ALIAN HAIDAR, LUKE KIRKBY
####    22964301, 22900426, 22885101
##### Due: 23 May, 2023, 11:59pm

## Overview
Automatic calibration of a holographic acquisition Rig. The purpose of this project is to implement the automatic calibration procedure for a holographic acquisition Rig. The inputs in play are a series of images taken by specially located cameras inside a room taking images of the same subject from different angles.

The program is written in Python 3.8.5 and uses the following libraries:
- OpenCV 4.5.1
- Numpy 1.19.5
- Matplotlib 3.3.4
- Pillow 8.1.0
- Scipy 1.6.0
- Scikit-Image 0.18.1
- PySimpleGUI 4.45.0

Main framework of the program's GUI is based on Qt5 and PySimpleGUI.

# How to Set Up

A virtual environment is required to run the project.\
To create a virtual environment, run the following command:
```
python3 -m venv virtual_venv | python -m venv virtual_venv
```
To install the required packages, run the following command:
```
pip install -r requirements.txt
```
To exit the virtual environment, run the following command:
```
deactivate
```
For Windows:
Set powershell permissions to allow all bypass
```
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force\
.\venv\bin\activate.ps1 | .\env\script\activate.ps1
```

# How to Run

To run the program, run the following command:
```
python3 main.py | python main.py
```

# How to Use

After all the required packages are installed, the program will run automatically when executed. Once ran, the program will gerate a graphical user interface in which the user can interact with. At first sight there are four buttons and three panels that the user can see. The four buttons are:\
**Load New Image** - This button allows the user to load the images that will be used for the calibration process.\
**Previous Image** - This button allows the user to scroll back to previous image.\
**Next Image** - This button allows the user to scroll forward to next image.\
**Delete Image** - This button allows the user to delete the selected image.\

The three panels are:\
**Perspective 0: Original Image** - Shows the original Image.\
**Perspective 0: Labeled HexaTargets** - Shows the location of the RGB dots on the screen.\
**3D Render** - This panel displays the 3D render of the selected images.\

When all images are selected successfully, it will ask the user to import the selected JSON files specific to the camera image used. This will import all calibration data, suchas the coordinates of the RGB dots, the camera matrix, the distortion coefficients, the rotation and translation vectors.\

Once all the images are loaded and the JSON files are imported, the user can now select the image that they want to calibrate. The user can scroll through the images using the **Previous Image** and **Next Image** buttons. The user can also delete the selected image using the **Delete Image** button.\

