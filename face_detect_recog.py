# Imports
from __future__ import print_function, division
import cv2
import os
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as image
from pattern_recog_func import *

# Functions
'''
    Returns: images and target array
    Function: reads in the images from the folder and loads them into 
              the images array and also while doing that makes the target
              array.
'''
def grab_images(folder):
    y = []
    images = []
    for filename in os.listdir(folder):
        if ('Gilbert' in filename):
            y.append(0)
        if ('Janet' in filename):
            y.append(1)
        if ('Luke' in filename):
            y.append(2)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images, y

'''
    Returns: the interpolated image
    Function: Finds the face in the single photo and then crops it and
              interpolates it.
'''
def crop_image(img):       
    cascPath = 'haarcascade_frontalface_default.xml'
    
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
                                         gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
     
    for (x, y, w, h) in faces:
        img = img[y: y+h , x :x+w]

    intp_img = interpol_im(img, dim1 = 45, dim2 = 60)
    return intp_img

'''
    Returns: a list of cropped faces and the coordinates for them
    Function: this function is **only** used to crop the image that
              has more than one face. This will take in one photo and return 
              a cropped photo for each face in a list.
'''
def crop_images(img):
    cascPath = 'haarcascade_frontalface_default.xml'
    
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
                                         gray,
                                         scaleFactor=1.3,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    

    images = []
    for (x, y, w, h) in faces:
        images.append(img[y: y+h , x :x+w])
        
    return images, faces

'''
    Returns: prediction
    Function: does the training with one missing index and then tests the
              trainer with the left out index. (for the leave-on-out test)
'''
def training(X, y, select_idx, n_comp = 50):
    Xtrain = np.delete(X, select_idx, axis = 0)
    ytrain = np.delete(y, select_idx)
    
    Xtest = X[select_idx].reshape(1, -1)
    ytest = y[select_idx]
    
    md_pca, Xtrain_proj = pca_X(Xtrain, n_comp = 50)
    Xtest_proj = md_pca.transform(Xtest)

    md_clf = svm_train(Xtrain_proj, ytrain)
    
    return md_clf.predict(Xtest_proj)

'''
    Returns: a list of the cropped images
    Function: this trains the whole data set and then loads in the whoswho image
              it then calls crop_images to get 3 separate cropped photos from the
              original photo. (one of gilbert, janet, luke). For each cropped
              image we call the pca_svm_pred on the image. Prints each image and
              prediction and the original image with the predictions next to their
              face.
'''
def whoswho(X, y, image_file):  
    # Train the whole entire data set !!
    md_pca, Xtrain_proj = pca_X(X, n_comp = 50)
    md_clf = svm_train(Xtrain_proj, y)
    
    # Load in test photo (whoswho.JPG) and crop them
    whoswho = image_file
    orig_im = image.imread(whoswho)
    images, faces = crop_images(orig_im)
    
    # goes through cropped images and called pca_svm_pred and for each prediction
    # it will print out the prediction made and add to the preds list.
    preds = []
    for i in range(len(images)):
        preds.append(pca_svm_pred(images[i], md_pca, md_clf))
        print("PCA+SVM predition for person", i, ":", names_dict.get(preds[i][0]))
      
    n_row = 1
    n_col = 3

    # this will plot each cropped image with a title of what the prediction was
    for i in range (n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i], cmap = plt.cm.gray)
        plt.title("Person" + str(i) + ": " + names_dict.get(preds[i][0]))
        plt.axis('off')
    

    # Create figure and show original image
    fig = plt.figure(figsize = (8, 8))
    ax = plt.subplot(111)
    ax.imshow(orig_im)
    plt.grid('off')
    plt.axis('off')
    
    # Set font
    font0 = FontProperties()
    font = font0.copy()
    font.set_style('italic')
    font.set_weight('bold')
    font.set_size('large')
    
    # This will put text (predictions) on the current plt image. 
    j = 0
    for (x, y, w, h) in faces:
        plt.text(x + 100, y, names_dict.get(preds[j][0]), color='pink', \
            size = 12, fontproperties = font)
        j+=1

    plt.show()
    
    return images


# **************** Main Program *********************

if __name__ == '__main__':

    '''
        b) Data set preparation
    '''
    # The dictionary
    names_dict = {0: 'Gilbert', 1: 'Janet', 2: 'Luke'}

    # Call grab_images to load the images from dir
    images, y = grab_images('training_pics/')

    # Crop the images and save them back into the images array
    for i in range(len(images)):
        images[i] = crop_image(images[i])

    X = np.vstack(images)


    '''
        c) Training and Validation
    '''
    # Goes through X and does a leave-one-out test to get the success
    # rate. 
    incorrect = 0

    for i in range(len(X)):
        y_pred = training(X, y, i)
        if (y[i] != y_pred[0]):
            incorrect += 1;

    success_rate = (len(X) - incorrect) / len(X) 
    print(success_rate)

    '''
        d) Testing
    '''
    images = whoswho(X, y, 'whoswho.JPG')










