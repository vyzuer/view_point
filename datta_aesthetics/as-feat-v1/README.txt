This directory contains accompanying code used in ACM MM 2013 Grand Challenge:
NHK Where is beauty? submission.

Behnaz Nojavan [behnaz.nojavan@gmail.com]
Subh Bhattacharya [subh@cs.ucf.edu]
07/11/2013

This release contains only frame based features that can be computed on single 
images. A separate release is being prepared for integrating these into videos
alongwith other features.

Following are the list of frame based  aesthetic features that are computed at 
cell level:

C1.Dark channel feature
C2.Luminosity feature
C3.S3 Sharpness
C4.Symmetry
C5.Low Depth of Field
C6.White Balance
C7.Colorfulness
C8.Color Harmony
C9.Eye Sensitivity

These are the aesthetic features that are computed at frame level:
F1. Normalized Area of dominant object
F2. Normalized distances of centroid of dominant objects wrt 4 stress points.

Usage:
[fa, fp, asf] = getAestheticFeatures(imread('img.png'));
where fa, fp are F1, F2, while asf is a structure containing C1-C9.
