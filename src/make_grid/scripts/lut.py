import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

lut = 255*np.ones((596, 596), dtype=int)
img = np.zeros((596,596), np.uint8)

##Contorno exterior
ext_cont = np.array([[27,42],[27,313],[42,327],[267,327],[267,552],[282,566],[553,566],
	[567,552],[567,284],[552,270],[327,270],[327,42],[313,27],[42,27]])
#cv.fillPoly(img, pts=[ext_cont], color=(0,255,0))
cv.fillPoly(img, pts=[ext_cont], color=(255,255,255))

##__Contorno interior__##
int_cont_1 = np.array([[270,270],[270,116],[239,85],[120,85],[89,116],[89,239],[120,269]])
cv.fillPoly(img, pts=[int_cont_1], color=(0,0,0))
int_cont_2 = np.array([[326,327],[326,476],[356,508],[474,508],[505,477],[505,358],[474,327]])
cv.fillPoly(img, pts=[int_cont_2], color=(0,0,0))
"""
for i in range(596): 
	for j in range(596):
		if(img[i,j]):
			img[i,j] = 255
			#print("Holi") 
			lut[i,j] = 0
np.save("LUT.npy", lut)
"""
fh_img = cv.flip(img, 1)
cv.imshow("image", fh_img)
cv.imwrite("lut.bmp",fh_img)
cv.waitKey(0)
