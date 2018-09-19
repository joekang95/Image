import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

img = misc.imread('image1.jpg')   

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]


gray = np.zeros((img.shape[0], img.shape[1]))
for row in range(len(img)):
    for col in range(len(img[row])):
        gray[row][col] = weightedAverage(img[row][col])
        
height = gray.shape[0]
width = gray.shape[1]

zero = np.zeros((height + 2, width + 2))
zero2 = np.zeros((height + 4, width + 4))

for i in range(0,height):
    for j in range(0,width):
        zero[i+1][j+1] = gray[i][j]
        
for i in range(0,height):
    for j in range(0,width):
        zero2[i+2][j+2] = gray[i][j]

n = gray.copy()
for i in range(1,height+1):
    for j in range(1,width+1):
        n[i-1][j-1] = (np.abs((zero[i-1][j-1] + 2*zero[i-1][j] + zero[i-1][j+1]) - (zero[i+1][j-1] + 2*zero[i+1][j] + zero[i+1][j+1])) +
                       np.abs((zero[i-1][j+1] + 2*zero[i][j+1] + zero[i+1][j]) - (zero[i-1][j-1] + 2*zero[i][j-1] + zero[i+1][j-1])))


b = gray.copy()
for i in range(1,height+1):
    for j in range(1,width+1):
        b[i-1][j-1] = (zero[i-1][j-1] + zero[i-1][j] + zero[i-1][j+1] +
                       zero[i][j-1] + zero[i][j] + zero[i][j+1] +
                       zero[i+1][j-1] + zero[i+1][j] + zero[i+1][j+1])/9

b2 = gray.copy()
for i in range(2,height+2):
    for j in range(2,width+2):
        b2[i-2][j-2] = (zero2[i-2][j-2] + zero2[i-2][j-1] + zero2[i-2][j] + zero2[i-2][j+1] + zero2[i-2][j+2] +
                        zero2[i-1][j-2] + zero2[i-1][j-1] + zero2[i-1][j] + zero2[i-1][j+1] + zero2[i-1][j+2] +
                        zero2[i][j-2] + zero2[i][j-1] + zero2[i][j] + zero2[i][j+1] + zero2[i][j+2] +
                        zero2[i+1][j-2] + zero2[i+1][j-1] + zero2[i+1][j] + zero2[i+1][j+1] + zero2[i+1][j+2] +
                        zero2[i+2][j-2] + zero2[i+2][j-1] + zero2[i+2][j] + zero2[i+2][j+1] + zero2[i+2][j+2] )/25



plt.figure(1)
### Gray ###
a = plt.subplot(2,2,1)
plt.imshow(gray, cmap = cm.Greys_r)
### Sobel ###
a = plt.subplot(2,2,2)
plt.imshow(n, cmap = cm.Greys_r)
### Blur 3*3 Average ###
a = plt.subplot(2,2,3)
plt.imshow(b, cmap = cm.Greys_r)
### Blur 5*5 Average ###
a = plt.subplot(2,2,4)
plt.imshow(b2, cmap = cm.Greys_r)
plt.show()
      
                    
                    
