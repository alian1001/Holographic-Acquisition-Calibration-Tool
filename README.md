# CITS4402 Computer Vision
## ERWIN BAUERNSCHMITT, ALIAN HAIDAR, LUKE KIRKBY
####    22964301, 22900426, 22885101
##### Due: 23 May, 2023, 11:59pm

## Overview
Automatic calibration of a holographic acquisition Rig. The purpose of this project is to implement the automatic calibration procedure for a holographic acquisition Rig. The inputs in play are a series of images taken by specially located cameras inside a room taking images of the same subject from different angles.

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