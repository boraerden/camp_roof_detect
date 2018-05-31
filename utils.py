import cv2 as cv
import numpy as np

def mask_to_contours(msk_path):
	mask = cv.imread(msk_path)
	mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
	boxes = cv.findContours(mask, mode=cv.cv.CV_RETR_LIST, method=cv.cv.CV_CHAIN_APPROX_SIMPLE)[0]
	return boxes

def widen_contours(contours, padding):
	wide_contours = []
	for contour in contours:
		if len(contour) != 4:
			continue

		topleft = contour[0][0]
		bottomleft = contour[1][0]
		bottomright = contour[2][0]
		topright = contour[3][0]

		newtopleft = [topleft[0]-padding, topleft[1]-padding]
		newtopright = [topright[0]+padding, topright[1]-padding]
		newbottomright = [bottomright[0]+padding, bottomright[1]+padding]
		newbottomleft = [bottomleft[0]-padding, bottomleft[1]+padding]

		wide_contour = np.array([[newtopleft],[newbottomleft],[newbottomright],[newtopright]], dtype=np.int32)

		wide_contours.append(np.asarray(wide_contour))
	np.asarray(wide_contours, dtype=np.int32)

	return wide_contours

def contours_to_xy_minmaxs(contours, size=256):
	xy_minmaxs = []
	for contour in contours:
		if len(contour) != 4:
			continue

		topleft = contour[0][0]
		bottomright = contour[2][0]

		xmin = max(0,topleft[0])
		xmax = min(size,bottomright[0])
		ymin = max(0,topleft[1])
		ymax = min(size,bottomright[1])

		xy_minmax = [xmin, ymin, xmax, ymax]
		xy_minmaxs.append(xy_minmax)

	return xy_minmaxs