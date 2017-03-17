# Imports
from __future__ import print_function, division
import matplotlib.pyplot as plt
import matplotlib.image as image
from pattern_recog_func import *
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn import svm


# **************** Main Program *********************
if __name__ == '__main__':

	'''
		a) Training and Validation
	'''
	# loading handwritten digits
	dig_data = load_digits()

	X = dig_data.data
	y = dig_data.target

	dig_img = dig_data.images

	# Training only on the first 60 images
	md_clf = svm_train(X[0:60], y[0:60])

	# Testing predictions on the next 20 images
	y_pred = md_clf.predict(X[60:81])
	
	success = 0	
	amount = 20
	incorrect = 0

	# This will be the new array for the targets that we just predicted on
	y_new = y[60:81]

	# Goes through the predictions we got back and compares against correct target
	for i in range (amount + 1):
	    if y_new[i] != y_pred[i]:
	        incorrect += 1
	        print("--------> index, actual digit, svm prediction: ", \
	        i, y_new[i], y_pred[i])
	success_rate = (amount - incorrect) / amount

	print("Total number of mid-identifications: ", incorrect)
	print("Success rate: ", success_rate)


	'''
		b) Testing
	'''
	# Load in the image and interpolate it and displaying it after interpolation
	unseen = image.imread("unseen_dig.png")
	interp_unseen = interpol_im(unseen, plot_new_im = True)

	# Displays the X[15] image which is also a "5", for human inspection
	plt.imshow(dig_img[15], cmap = 'binary', interpolation = "nearest")
	plt.show()

	# Rescales the interpolated image and then predicts rescaled and un-rescaled
	rescale_unseen = rescale_pixel(interp_unseen)
	pred_interp = md_clf.predict(interp_unseen.reshape(1, -1))
	pred_rescale =md_clf.predict(rescale_unseen.reshape(1, -1))

	print("Un-rescaled prediction (should be 5): ", pred_interp[0])
	print("Rescaled prediction (should be 5): ", pred_rescale[0])

