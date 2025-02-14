#!/bin/bash

# Install required Python packages
pip install dlib deepface fer opencv-python

# Download dlib's face landmark predictor
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat /home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/utils/
