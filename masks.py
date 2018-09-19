import numpy as np
from scipy import misc
from scipy.ndimage.filters import generic_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statistics


img = misc.imread('image1.jpg')
img2 = misc.imread('noise.jpg')
img3 = misc.imread('max.jpg')
img4 = misc.imread('min.jpg')
img5 = misc.imread('image3.jpg')
img6 = misc.imread('image2.jpg')
img7 = misc.imread('image4.jpg')

def low_pass_average(P):
    return (P[0] + P[1] + P[2] + P[3] + P[4] + P[5] + P[6] + P[7] + P[8])/9

def low_pass_average2(P):
    a = 0
    for i in range(0,25):
        a += P[i]
    a = a/25
    return a

def low_pass_average3(P):
    a = 0
    for i in range(0,81):
        a += P[i]
    a = a/81
    return a

def low_pass_average4(P):
    a = 0
    for i in range(0,225):
        a += P[i]
    a = a/225
    return a

def low_pass_weighted(P):
    return (P[0] + 2*P[1] + P[2] + 2*P[3] + 4*P[4] + 2*P[5] + P[6] + 2*P[7] + P[8])/16

def low_pass_weighted2(P):
    return (P[0] + 4*P[1] + 6*P[2] + 4*P[3] + P[4] +
            4*P[5] + 16*P[6] + 24*P[7] + 16*P[8] + 4*P[9] +
            6*P[10] + 24*P[11] + 36*P[12] + 24*P[13] + 6*P[14] +
            4*P[15] + 16*P[16] + 24*P[17] + 16*P[18] + 4*P[19] +
            P[20] + 4*P[21] + 6*P[22] + 4*P[23] + P[24])/256

def mid_pass(P):
    return statistics.median([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8]])

def max_filter(P):
    return max([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8]])

def min_filter(P):
    return min([P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8]])

def high_pass(P):
    return np.abs(8*P[4] - (P[0] + P[1] + P[2] + P[3] + P[5] + P[6] + P[7] + P[8]))/9 

def high_pass2(P):
    return np.abs(12*P[4] - (P[0] + 2*P[1] + P[2] + 2*P[3] + 2*P[5] + P[6] + 2*P[7] + P[8]))/16

def one_filter(P):
    return np.abs(2*P[4] - (P[3] + P[7]))

def laplace_filter(P):
    return np.abs(5*(P[4] - (P[1] + P[3] + P[4] + P[5] + P[7])/5))

def robert_cross(P):
    return(np.abs(P[0] - P[3]) + np.abs(P[1] - P[2]))

def robert_cross2(P):
    return(np.abs(P[4] - P[6]) + np.abs(P[4] - P[8]))

def prewitt_filter(P):
    return (np.abs((P[0] + P[1] + P[2]) - (P[6] + P[7] + P[8])) +
            np.abs((P[2] + P[5] + P[8]) - (P[0] + P[3] + P[6])))

def prewitt_filter2(P):
    return (np.abs(-3*(P[0] + P[4] + P[8] + P[12]) - (P[1] + P[5] + P[9] + P[13]) + (P[2] + P[6] + P[10] + P[14]) + 3*(P[3] +P[7] + P[11] + P[15])) +
            np.abs(3*(P[0] + P[1] + P[2] + P[3]) + (P[4] + P[5] + P[6] + P[7]) - (P[8] + P[9] + P[10] + P[11]) - 3*(P[12] + P[13] + P[14] + P[15])))

def prewitt_filter3(P):
    return (np.abs(-2*(P[0] + P[5] + P[10] + P[15] + P[20]) - (P[1] + P[6] + P[11] + P[16] + P[21]) + (P[3] + P[8] + P[13] + P[18] + P[23]) + 2*(P[4] +P[9] + P[14] + P[19] + P[24])) +
            np.abs(2*(P[0] + P[1] + P[2] + P[3] + P[4]) + (P[5] + P[6] + P[7] + P[8] + P[9]) - (P[15] + P[16] + P[17] + P[18] + P[19]) - 2*(P[20] + P[21] + P[22] + P[23] + P[24])))

def sobel_filter(P):
    return (np.abs((P[0] + 2*P[1] + P[2]) - (P[6] + 2*P[7] + P[8])) +
            np.abs((P[2] + 2*P[5] + P[8]) - (P[0] + 2*P[3] + P[6])))

def frei_filter(P):
    return (np.abs((P[0] + 1.414 * P[1] + P[2]) - (P[6] + 1.414 * P[7] + P[8])) +
            np.abs((P[2] + 1.414 * P[5] + P[8]) - (P[0] + 1.414 * P[3] + P[6])))

def sharr_filter(P):
    return (np.abs((3*P[0] + 10*P[1] + 3*P[2]) - (3*P[6] + 10*P[7] + 3*P[8])) +
            np.abs((3*P[2] + 10*P[5] + 3*P[8]) - (3*P[0] + 10*P[3] + 3*P[6])))

def log_filter(P):
    return (-5*(P[2] + P[6] + P[8] + P[10] + P[14] + P[16] + P[18] + P[22]) +
                  -2*(P[7] + P[11] + P[13] + P[17]) + 16*P[12])
def log_filter2(P):
    return ((P[1] + P[2] + 2*(P[3] + P[4] + P[5]) + P[6] + P[7]) +
            (P[9] + 2*P[10] + 4*P[11] + 5*(P[12] + P[13] + P[14]) + 4*P[15] + 2*P[16] +P[17]) +
            (P[18] + 4*P[19] + 5*P[20] + 3*P[21] + 3*P[23] + 5*P[24] + 4*P[25] + P[26]) +
            (2*P[27] + 5*P[28] + 3*P[29] - 12*P[30] - 24*P[31] - 12*P[32] + 3*P[33] + 5*P[34] + 2*P[35]) +
            (2*P[36] + 5*P[37] - 24*P[39] - 40*P[40] - 24*P[41] + 5*P[43] + 2*P[44]) +
            (2*P[45] + 5*P[46] + 3*P[47] - 12*P[49] - 24*P[49] - 12*P[50] + 3*P[51] + 5*P[52] + 2*P[53]) +
            (P[54] + 4*P[55] + 5*P[56] + 3*P[57] + 3*P[59] + 5*P[60] + 4*P[61] + P[62]) +
            (P[63] + 2*P[64] + 4*P[65] + 5*(P[66] + P[67] + P[68]) + 4*P[69] + 2*P[70] +P[71]) +
            (P[73] + P[74] + 2*(P[75] + P[76] + P[77]) + P[78] + P[79]))

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

gray = np.zeros((img.shape[0], img.shape[1]))
for row in range(len(img)):
    for col in range(len(img[row])):
        gray[row][col] = weightedAverage(img[row][col])

gray2 = np.zeros((img2.shape[0], img2.shape[1]))
gray3 = img3
gray4 = img4
gray5 = np.zeros((img5.shape[0], img5.shape[1]))
gray6 = np.zeros((img6.shape[0], img6.shape[1]))
gray7 = np.zeros((img7.shape[0], img7.shape[1]))
for row in range(len(img2)):
    for col in range(len(img2[row])):
        gray2[row][col] = weightedAverage(img2[row][col])

for row in range(len(img5)):
    for col in range(len(img5[row])):
        gray5[row][col] = weightedAverage(img5[row][col])

for row in range(len(img6)):
    for col in range(len(img6[row])):
        gray6[row][col] = weightedAverage(img6[row][col])

for row in range(len(img7)):
    for col in range(len(img7[row])):
        gray7[row][col] = weightedAverage(img7[row][col])


  
misc.imsave('image1-gray.jpg', gray)
misc.imsave('image3-gray.jpg', gray5)
misc.imsave('image2-gray.jpg', gray6)
misc.imsave('image4-gray.jpg', gray7)



### gray = image1.jpg ###

low = generic_filter(gray, low_pass_average, (3, 3))
misc.imsave('image1-low_pass_average.jpg',low)

low2 = generic_filter(gray, low_pass_average2, (5, 5))
misc.imsave('image1-low_pass_average2.jpg',low2)

low3 = generic_filter(gray, low_pass_average3, (9, 9))
misc.imsave('image1-low_pass_average3.jpg',low3)

low4 = generic_filter(gray, low_pass_average4, (15, 15))
misc.imsave('image1-low_pass_average4.jpg',low4)

lowg = generic_filter(gray, low_pass_weighted, (3, 3))
misc.imsave('image1-low_pass_weighted.jpg',lowg)

lowg2 = generic_filter(gray, low_pass_weighted2, (5, 5))
misc.imsave('image1-low_pass_weighted2.jpg',lowg2)

lowg3 = generic_filter(lowg, low_pass_weighted, (3, 3))
lowg3 = generic_filter(lowg3, low_pass_weighted, (3, 3))
lowg3 = generic_filter(lowg3, low_pass_weighted, (3, 3))
misc.imsave('image1-low_pass_weighted3.jpg',lowg3)

lowg4 = generic_filter(lowg3, low_pass_weighted, (3, 3))
lowg4 = generic_filter(lowg4, low_pass_weighted, (3, 3))
lowg4 = generic_filter(lowg4, low_pass_weighted, (3, 3))
misc.imsave('image1-low_pass_weighted4.jpg',lowg4)


### gray2 = noise.jpg, gray3 = max.jpg, gray4 = min.jpg###

mid = generic_filter(gray2, mid_pass, (3, 3))
misc.imsave('noise-mid.jpg', mid)

maxf = generic_filter(gray4, max_filter, (3, 3))
#maxf = generic_filter(maxf, max_filter, (3, 3))
misc.imsave('min-max.jpg', maxf)

maxf = generic_filter(maxf, max_filter, (3, 3))
misc.imsave('min-max2.jpg', maxf )

minf = generic_filter(gray3, min_filter, (3, 3))
#minf = generic_filter(minf, min_filter, (3, 3))
misc.imsave('max-min.jpg', minf)

minf = generic_filter(minf, min_filter, (3, 3))
misc.imsave('max-min2.jpg', minf)


### gray5 = image3.jpg ###

mid = generic_filter(gray5, mid_pass, (3, 3))
misc.imsave('image3-mid.jpg', mid)

minf = generic_filter(gray5, min_filter, (3, 3))
misc.imsave('image3-min2.jpg', minf)

low5 = generic_filter(gray5, low_pass_weighted, (3, 3))
low5 = generic_filter(low5, low_pass_weighted, (3, 3))
low5 = generic_filter(low5, low_pass_weighted, (3, 3))
misc.imsave('image3-low_pass_average5.jpg', low5)


### gray = image1.jpg ###

high = generic_filter(gray, high_pass, (3, 3))
misc.imsave('image1-high1-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image1-high2-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image1-high3-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image1-high4-abs.jpg', high)

high2 = generic_filter(gray, high_pass2, (3, 3))
misc.imsave('image1-high-average1-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image1-high-average2-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image1-high-average3-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image1-high-average4-abs.jpg', high2)

### gray = image1.jpg ###

one = generic_filter(gray, one_filter, (3, 3))
misc.imsave('image1-one.jpg',one)

laplace = generic_filter(gray, laplace_filter, (3, 3))
misc.imsave('image1-laplace.jpg',laplace)

robert = generic_filter(gray, robert_cross, (2, 2))
misc.imsave('image1-robert.jpg',robert)

robert2 = generic_filter(gray, robert_cross2, (3, 3))
misc.imsave('image1-robert2.jpg',robert2)

prewitt = generic_filter(gray, prewitt_filter, (3, 3))
misc.imsave('image1-prewitt.jpg',prewitt)

prewitt2 = generic_filter(gray, prewitt_filter2, (4, 4))
misc.imsave('image1-prewitt2.jpg',prewitt2)

prewitt3 = generic_filter(gray, prewitt_filter3, (5, 5))
misc.imsave('image1-prewitt3.jpg',prewitt3)                   

sobel = generic_filter(gray, sobel_filter, (3, 3))
misc.imsave('image1-sobel.jpg',sobel)

frei = generic_filter(gray, frei_filter, (3, 3))
misc.imsave('image1-frei.jpg',frei)

sharr = generic_filter(gray, sharr_filter, (3, 3))
misc.imsave('image1-sharr.jpg',sharr)

log = generic_filter(gray, log_filter, (5, 5))
misc.imsave('image1-log.jpg',log)

log2 = generic_filter(gray, log_filter2, (9, 9))
misc.imsave('image1-log2.jpg',log2)




### gray6 = image2.jpg ###

low = generic_filter(gray6, low_pass_average, (3, 3))
misc.imsave('image2-low_pass_average.jpg',low)

low2 = generic_filter(gray6, low_pass_average2, (5, 5))
misc.imsave('image2-low_pass_average2.jpg',low2)

low3 = generic_filter(gray6, low_pass_average3, (9, 9))
misc.imsave('image2-low_pass_average3.jpg',low3)

low4 = generic_filter(gray6, low_pass_average4, (15, 15))
misc.imsave('image2-low_pass_average4.jpg',low4)

lowg = generic_filter(gray6, low_pass_weighted, (3, 3))
misc.imsave('image2-low_pass_weighted.jpg',lowg)

lowg2 = generic_filter(gray6, low_pass_weighted2, (5, 5))
misc.imsave('image2-low_pass_weighted2.jpg',lowg2)

lowg3 = generic_filter(lowg, low_pass_weighted, (3, 3))
lowg3 = generic_filter(lowg3, low_pass_weighted, (3, 3))
lowg3 = generic_filter(lowg3, low_pass_weighted, (3, 3))
misc.imsave('image2-low_pass_weighted3.jpg',lowg3)

lowg4 = generic_filter(lowg3, low_pass_weighted, (3, 3))
lowg4 = generic_filter(lowg4, low_pass_weighted, (3, 3))
lowg4 = generic_filter(lowg4, low_pass_weighted, (3, 3))
misc.imsave('image2-low_pass_weighted4.jpg',lowg4)

high = generic_filter(gray6, high_pass, (3, 3))
misc.imsave('image2-high1-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image2-high2-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image2-high3-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image2-high4-abs.jpg', high)

high2 = generic_filter(gray6, high_pass2, (3, 3))
misc.imsave('image2-high-average1-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image2-high-average2-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image2-high-average3-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image2-high-average4-abs.jpg', high2)

one = generic_filter(gray6, one_filter, (3, 3))
misc.imsave('image2-one.jpg',one)

laplace = generic_filter(gray6, laplace_filter, (3, 3))
misc.imsave('image2-laplace.jpg',laplace)

robert = generic_filter(gray6, robert_cross, (2, 2))
misc.imsave('image2-robert.jpg',robert)

robert2 = generic_filter(gray6, robert_cross2, (3, 3))
misc.imsave('image2-robert2.jpg',robert2)

prewitt = generic_filter(gray6, prewitt_filter, (3, 3))
misc.imsave('image2-prewitt.jpg',prewitt)

prewitt2 = generic_filter(gray6, prewitt_filter2, (4, 4))
misc.imsave('image2-prewitt2.jpg',prewitt2)

prewitt3 = generic_filter(gray6, prewitt_filter3, (5, 5))
misc.imsave('image2-prewitt3.jpg',prewitt3)                   

sobel = generic_filter(gray6, sobel_filter, (3, 3))
misc.imsave('image2-sobel.jpg',sobel)

frei = generic_filter(gray6, frei_filter, (3, 3))
misc.imsave('image2-frei.jpg',frei)

sharr = generic_filter(gray6, sharr_filter, (3, 3))
misc.imsave('image2-sharr.jpg',sharr)

log = generic_filter(gray6, log_filter, (5, 5))
misc.imsave('image2-log.jpg',log)

log2 = generic_filter(gray6, log_filter2, (9, 9))
misc.imsave('image2-log2.jpg',log2)


### gray7 = image4.jpg ###

low = generic_filter(gray7, low_pass_average, (3, 3))
misc.imsave('image4-low_pass_average.jpg',low)

low2 = generic_filter(gray7, low_pass_average2, (5, 5))
misc.imsave('image4-low_pass_average2.jpg',low2)

low3 = generic_filter(gray7, low_pass_average3, (9, 9))
misc.imsave('image4-low_pass_average3.jpg',low3)

low4 = generic_filter(gray7, low_pass_average4, (15, 15))
misc.imsave('image4-low_pass_average4.jpg',low4)

lowg = generic_filter(gray7, low_pass_weighted, (3, 3))
misc.imsave('image4-low_pass_weighted.jpg',lowg)

lowg2 = generic_filter(gray7, low_pass_weighted2, (5, 5))
misc.imsave('image4-low_pass_weighted2.jpg',lowg2)

lowg3 = generic_filter(lowg, low_pass_weighted, (3, 3))
lowg3 = generic_filter(lowg3, low_pass_weighted, (3, 3))
lowg3 = generic_filter(lowg3, low_pass_weighted, (3, 3))
misc.imsave('image4-low_pass_weighted3.jpg',lowg3)

lowg4 = generic_filter(lowg3, low_pass_weighted, (3, 3))
lowg4 = generic_filter(lowg4, low_pass_weighted, (3, 3))
lowg4 = generic_filter(lowg4, low_pass_weighted, (3, 3))
misc.imsave('image4-low_pass_weighted4.jpg',lowg4)

high = generic_filter(gray7, high_pass, (3, 3))
misc.imsave('image4-high1-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image4-high2-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image4-high3-abs.jpg', high)

high = generic_filter(high, high_pass, (3, 3))
misc.imsave('image4-high4-abs.jpg', high)

high2 = generic_filter(gray7, high_pass2, (3, 3))
misc.imsave('image4-high-average1-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image4-high-average2-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image4-high-average3-abs.jpg', high2)

high2 = generic_filter(high2, high_pass2, (3, 3))
misc.imsave('image4-high-average4-abs.jpg', high2)

one = generic_filter(gray7, one_filter, (3, 3))
misc.imsave('image4-one.jpg',one)

laplace = generic_filter(gray7, laplace_filter, (3, 3))
misc.imsave('image4-laplace.jpg',laplace)

robert = generic_filter(gray7, robert_cross, (2, 2))
misc.imsave('image4-robert.jpg',robert)

robert2 = generic_filter(gray7, robert_cross2, (3, 3))
misc.imsave('image4-robert2.jpg',robert2)

prewitt = generic_filter(gray7, prewitt_filter, (3, 3))
misc.imsave('image4-prewitt.jpg',prewitt)

prewitt2 = generic_filter(gray7, prewitt_filter2, (4, 4))
misc.imsave('image4-prewitt2.jpg',prewitt2)

prewitt3 = generic_filter(gray7, prewitt_filter3, (5, 5))
misc.imsave('image4-prewitt3.jpg',prewitt3)                   

sobel = generic_filter(gray7, sobel_filter, (3, 3))
misc.imsave('image4-sobel.jpg',sobel)

frei = generic_filter(gray7, frei_filter, (3, 3))
misc.imsave('image4-frei.jpg',frei)

sharr = generic_filter(gray7, sharr_filter, (3, 3))
misc.imsave('image4-sharr.jpg',sharr)

log = generic_filter(gray7, log_filter, (5, 5))
misc.imsave('image4-log.jpg',log)

log2 = generic_filter(gray7, log_filter2, (9, 9))
misc.imsave('image4-log2.jpg',log2)
