# Imports
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline
from sklearn import svm
from sklearn.decomposition import PCA


# Functions
'''
	Returns: flattened interpolated image
	Function: interpolates the image given onto a grid with the dim parameters
'''
def interpol_im(im, dim1 = 8, dim2 = 8, plot_new_im = False, cmap = 'binary', \
	axis_off = False, grid_off = False):
	
	if len(im.shape) == 3:
		im = im[:,:, 0]

	x = np.arange(im.shape[1])
	y = np.arange(im.shape[0])

	f2d = interp2d(x, y, im)

	x_new = np.linspace(0, im.shape[1], dim1)
	y_new = np.linspace(0, im.shape[0], dim2)

	im_new = f2d(x_new, y_new)

	if(plot_new_im):
		if (axis_off):
			plt.axis("off")
		if (grid_off):
			plt.grid("off")
		plt.imshow(im_new, cmap = cmap, interpolation = "nearest")
        plt.show()

	return im_new.flatten()

'''
	Returns: the prediction
	Function: takes in the image file, interpolates the image and makes a 
			  predition using the passed in md_clf
'''
def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60):
 	interp_im = interpol_im(imfile, dim1 = dim1, dim2 = dim2)
 	X_proj = md_pca.transform(interp_im.reshape(1, -1))
 	prediction = md_clf.predict(X_proj.reshape(1, -1))

 	return prediction

'''
	Returns: md_pca, X_proj
	Function: takes in X, the data, and gets the PCA components and does a fit
			  and transform on X. 
'''
def pca_X(X, n_comp = 10):
 	md_pca = PCA(n_comp, whiten = True)
 	md_pca.fit(X)
 	X_proj = md_pca.transform(X)

 	return md_pca, X_proj

'''
	Returns: the new rescaled image
	Function: rescales the values of the inputted image to a pixel range from
			  0-15 and then reverts the image and returns.
''' 
def rescale_pixel(unseen):
    unseen = unseen * 15
    unseen = unseen.astype(int)

    for i in range(len(unseen)):
        if (unseen[i] == 0):
            unseen[i] = 15
        else:
            unseen[i] = 0

    return unseen.reshape(8, 8)

'''
	Returns: md_clf
	Function: svm training on X and returns the trainer.
'''
def svm_train(X, y, gamma = 0.001, C = 100):
	md_clf = svm.SVC(gamma=0.001, C=100.)
	md_clf.fit(X, y)

	return md_clf































