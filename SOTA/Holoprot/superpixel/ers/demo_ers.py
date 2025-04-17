from ers import ERS
import cv2
import numpy as np

def colormap(input,colors):
	height = input.shape[0]
	width  = input.shape[1]
	output = np.zeros([height, width, 3], np.uint8)
	for y in range(0, height):
		for x in range(0, width):
			id = int(input[y, x])
			for k in range(0, 3):
				output[y,x,k] = colors[id, k]
	return output

nC = 20
img = cv2.imread("../images/242078.jpg")

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

e = ERS(0.5, 5.0)
segmentation = e.computeSegmentation(np.uint8(grayImg), nC)

colors = np.uint8(np.random.rand(nC, 3) * 255)
output = colormap(segmentation, colors)
cv2.imshow("img", img)
cv2.imshow("segmentation",output)
cv2.waitKey()
cv2.destroyAllWindows()
