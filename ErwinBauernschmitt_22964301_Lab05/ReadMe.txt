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