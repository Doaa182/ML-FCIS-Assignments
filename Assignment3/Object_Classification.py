from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.svm import SVC
import numpy as np
import cv2
import os

##############################################################################################################################################################

# outer folder_name of train
train_folder_name = "train"

# list of classes_names of train
# train_classes_names = ["accordian", "dollar_bill", "motorbike", "Soccer_ball"]
train_classes_names = os.listdir(train_folder_name)

# empty list to store the imgs_paths, imgs and labels for train
train_imgs_paths = []
train_imgs = []
train_labels = []
        
# iterate over each class
for train_class_name in train_classes_names:
    train_class_path = os.path.join(train_folder_name, train_class_name)  #  path of the class folder
    train_files_names = os.listdir(train_class_path)
    
    # iterate over each img in the folder
    for train_file_name in train_files_names :
        train_img_path = os.path.join(train_class_path, train_file_name)  # path of the img file
        train_imgs_paths.append(train_img_path)  # Append the img path to the list
        train_img = cv2.imread(train_img_path) # Read image
        train_imgs.append(train_img)
        train_labels.append(train_class_name)
        
        
###############################################################################        

# outer folder_name of test
test_folder_name = "test"

# list of classes_names of test
# test_classes_names = ["accordian", "dollar_bill", "motorbike", "Soccer_ball"]
test_classes_names = os.listdir(test_folder_name)

# empty list to store the imgs_paths, imgs and labels for test
test_imgs_paths = []
test_imgs = []
test_labels = []

# iterate over each class
for test_class_name in test_classes_names:
    test_class_path = os.path.join(test_folder_name, test_class_name)  #  path of the class folder
    test_files_names = os.listdir(test_class_path)
    
    # iterate over each img in the folder
    for test_file_name in test_files_names :
        test_img_path = os.path.join(test_class_path, test_file_name)  # path of the img file
        test_imgs_paths.append(test_img_path)  # Append the img path to the list
        test_img = cv2.imread(test_img_path) # Read image
        test_imgs.append(test_img)
        test_labels.append(test_class_name)        
        
##############################################################################################################################################################
    
# extract hog features for train
train_hog_features = []

for train_img in train_imgs:
    train_resized_img = cv2.resize(train_img, (128,64))
    train_fd, train_hog_image = hog(train_resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    train_hog_features.append(train_fd)
    
###############################################################################

# extract hog features for test
test_hog_features = []

for test_img in test_imgs:
    test_resized_img = cv2.resize(test_img, (128,64))
    test_fd, test_hog_image = hog(test_resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    test_hog_features.append(test_fd)

##############################################################################################################################################################

# convert feature and label lists to np.array
X_train = np.array(train_hog_features)
X_test = np.array(test_hog_features)
Y_train = np.array(train_labels)
Y_test = np.array(test_labels)

###############################################################################

# train SVM using linear
clf = SVC(kernel='linear')
clf = clf.fit(X_train, Y_train)

# predictions on test 
Y_pred = clf.predict(X_test)

# calculate accuracy 
SVM_accuracy = accuracy_score(Y_test, Y_pred)
print ("SVM Classifier Accuracy (Linear) : ", SVM_accuracy)

###############################################################################

# train SVM using poly 
clf = SVC(kernel='poly', degree=3)
clf = clf.fit(X_train, Y_train)

# predictions on test 
Y_pred = clf.predict(X_test)

# calculate accuracy 
SVM_accuracy = accuracy_score(Y_test, Y_pred)
print ("SVM Classifier Accuracy (Poly) : ", SVM_accuracy)

###############################################################################

# train SVM using rbf 
clf = clf = SVC(kernel='rbf')
clf = clf.fit(X_train, Y_train)

# predictions on test 
Y_pred = clf.predict(X_test)

# calculate accuracy 
SVM_accuracy = accuracy_score(Y_test, Y_pred)
print ("SVM Classifier Accuracy (RBF): ", SVM_accuracy)

##############################################################################################################################################################










