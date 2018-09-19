import numpy as np
from scipy import misc
from scipy.ndimage.filters import generic_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statistics

img = misc.imread('noise.jpg')

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def mid_pass(P):
    return statistics.median([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8]])


gray2 = np.zeros((img.shape[0], img.shape[1]))
for row in range(len(img)):
    for col in range(len(img[row])):
        gray2[row][col] = weightedAverage(img[row][col])

gray = generic_filter(gray2, mid_pass, (3, 3))

N = 3
Width = img.shape[0]
Height = img.shape[1]
xc = Width // 2 
yc = Height // 2 
for level in range(N):
    for j in range(Height):
        line = [0]*Width
        for i in range(0,Width,2):
            gp1 = gray[i][j]
            gp2 = gray[i+1][j]
            average = round((gp1 + gp2) / 2 )
            diff = round((gp1 - gp2) / 2 )
            k = i // 2
            line[k] = average
            line[xc+k] = diff + 128
        for i in range(Width):
            gray[i][j] = line[i]

    for i in range(Width):
        line = [0]*Height
        for j in range(0,Height,2):
            gp1 = gray[i][j]
            gp2 = gray[i][j+1]
            average = round((gp1 + gp2) / 2 )
            diff = round((gp1 - gp2) / 2 )
            k = j // 2
            line[k] = average
            line[yc+k] = diff + 128
        for j in range(Height):
            gray[i][j] = line[j]

    Width = Width // 2
    Height = Height // 2
    xc = Width // 2 
    yc = Height // 2 
        
misc.imsave('harr3.jpg', gray)
