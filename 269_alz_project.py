import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
import pickle
from PIL import Image
import time
import os
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from evaluate_most_confused_labels import evaluate_most_confused_labels
from data_selector import data_selector
from fix_labels import fix_labels
from sklearn.svm import SVC
import subprocess
import shutil

def create_ImageDataGenerator(class_1, class_2, data_path):
    if (class_1 not in all_possible_classes) or (class_2 not in all_possible_classes):
        raise Exception("Invalid Classes! Valid classes are: " + str(all_possible_classes))
    
    if class_1.lower() == class_2.lower():
        raise Exception("Both Classes are the same!")
        
    class_1_path = data_path + class_1 
    class_2_path = data_path + class_2
    
    merged_path = './ALZ_Combine/'
    
    ## write an rsync commands to merge the directories
    rsync_cmd = 'rsync' + ' -avzh ' + class_1_path + ' ' + class_2_path + ' ' + merged_path

    print(class_1_path)
    print(class_2_path)

    ## run the rsync command
    subprocess.run(rsync_cmd, shell=True)
    
    
    train_dr = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,fill_mode='constant',cval=0,
                                                           brightness_range=bright_range,zoom_range=zoom,
                                                           data_format='channels_last',zca_whitening=False)

    train_data_gen = train_dr.flow_from_directory(directory=merged_path,target_size=dim,
                                              batch_size=5000)
        
    
    return train_data_gen


def getTieredData(class_1, class_2, path, idxStart, idxEnd):
    
    # get train data
    data_gen = create_ImageDataGenerator(class_1,class_2, path)
    im_data,im_labels =  data_gen.next()
    #print(train_data[1,:,:,:].shape)
    
    model = SVC(kernel='linear')
    d1, d2, d3, d4 = im_data[:,:,:,:].shape
    im_data_reshaped = im_data.reshape((d1, d2*d3*d4))
    #print(train_data_reshaped.shape)
    
    im_class_labels = np.argmax(im_labels, axis=1)
    model.fit(im_data_reshaped[:550,:],im_class_labels[:550])
    mly = model.decision_function(im_data_reshaped[idxStart:idxEnd,:])

    new_im_class_labels = []
    d = {'MildDemented':0, 'ModerateDemented':1, 'NonDemented':2, 'VeryMildDemented':3}
    for x in im_class_labels:
      if x == 0:
        one_hot_encode = [0,0,0,0]
        one_hot_encode[d[class_1]] = 1
        new_im_class_labels.append(one_hot_encode)
      elif x == 1:
        one_hot_encode = [0,0,0,0]
        one_hot_encode[d[class_2]] = 1
        new_im_class_labels.append(one_hot_encode)

    myl_train = list(map(list, zip(abs(mly), new_im_class_labels[idxStart:idxEnd],range(idxStart,idxEnd))))

    # get easy, med, hard data & labels
    easy_data, med_data, hard_data = data_selector(np.array(myl_train,dtype=object))

    easyIdx = easy_data[:,2].astype(int)
    easy_data_im = im_data[easyIdx,:,:,:]
    easy_data_labels = easy_data[:,1]
    print(easy_data.shape)
    print(easy_data_im.shape)
    print(easy_data_labels.shape)
    
    medIdx = med_data[:,2].astype(int)
    med_data_im = train_data[medIdx,:,:,:]
    med_data_labels = med_data[:,1]
    print(med_data.shape)
    print(med_data_im.shape)
    print(med_data_labels.shape)
    
    hardIdx = hard_data[:,2].astype(int)
    hard_data_im = train_data[hardIdx,:,:,:]
    hard_data_labels = hard_data[:,1]
    print(hard_data.shape)
    print(hard_data_im.shape)
    print(hard_data_labels.shape)

    dir_path = './ALZ_Combine/'
    
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
        
    return easy_data_im, easy_data_labels, med_data_im, med_data_labels, hard_data_im, hard_data_labels



dirname = os.path.dirname(__file__)

train_data_path =  os.path.join(dirname, 'data/train/')
test_data_path = os.path.join(dirname, 'data/test/')
model_path = os.path.join(dirname, 'val20_epochs10_START_testacc82.h5')

start = time.time()

## variables for starting model (x% val_acc with y val_loss)

dim = (176,208)               # input image dimensions                                              (176,208)
test_split_percent = .1       # % of total data for testing                                         .1
validation_split_percent = .2 # % of total data for validation                                      .2
zoom = [.99,1.01]             # zoom range (for a fixed zoom put the same value in both parameters) [.8,.8]
bright_range = [.8,1.2]       # brightness range                                                    [.8,1.2] 
layers_unlocked = False       # unlock the imported pre-training layers?                            False  
lr = 0.0001                   # learning rate for optimizer                                         0.0001
batch = 20                    # batch size for model fitting                                        20
eps = 10                      # number of epochs to run                                             100
momentum = .9                 # momentum of SGD                                                     .9

save_model_name = "val%2d_epochs%d_START"%(validation_split_percent*100,eps)   # automatically generate a model save name
print(save_model_name)

# This section uses the ImageDataGenerator and flow_from_directory functions to sort the images by label
#actual dimensions 176x208x1

# introduced zoom, blew up the image of the brain itself, and brightness range to adjust for different brightness
train_dr = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,fill_mode='constant',cval=0,
                                                           brightness_range=bright_range,zoom_range=zoom,
                                                           data_format='channels_last',zca_whitening=False)

#CHANGE PATH TO TRAINING DATASET HERE
train_data_gen = train_dr.flow_from_directory(directory=train_data_path,target_size=dim,
                                              batch_size=5000)

# Change to zoom = [1,1] to use normal test data
#CHANGE PATH TO TEST DATASET HERE
test_dr = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,fill_mode='constant',cval=0,zoom_range=[1,1],
                                                          data_format='channels_last') 
test_data_gen = test_dr.flow_from_directory(directory=test_data_path,target_size=dim,batch_size=5000,
                                           shuffle = False) # test data should not be shuffled to keep labels

# This section assigns the images to numpy arrays for the data and labels, takes a long time on google Colab
# EX: train_data = numpy array of image data, train_labels = numpy array of labels 

train_data,train_labels =  train_data_gen.next()
test_data,test_labels = test_data_gen.next()

# cocatenate arrays, combining all data 
# test data and training data are not from the same part of the brain so we need to combine the data 
# need to join the entire data 
# training got one part of brain and test got another 
total_data = np.concatenate((train_data,test_data))
total_labels = np.concatenate((train_labels,test_labels))
print(total_data.shape)
print(total_labels.shape)

# train test split
# split the combined data set into two different random piles

initial_split = test_split_percent+validation_split_percent
test_val_split = test_split_percent/initial_split

# split into training and (test + validation)
train_data, test_val_data, train_labels, test_val_labels = train_test_split(total_data,total_labels,
                                                                            test_size=initial_split)

# split (test + validation) into test and validation sets
test_data, val_data, test_labels, val_labels = train_test_split(test_val_data,test_val_labels,
                                                                test_size=test_val_split)

train_data_total = train_data
train_labels_total = train_labels
test_data_total = test_data
test_labels_total = test_labels

# Check array dimensions
print(train_data.shape)
print(train_labels.shape)
print(val_data.shape)
print(val_labels.shape)
print(test_data.shape)
print(test_labels.shape)


vg_model = load_model(model_path)

# compile
opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True,name='SGD') 
vg_model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

train_scores = vg_model.evaluate(train_data, train_labels)
val_scores = vg_model.evaluate(val_data,val_labels)
test_scores = vg_model.evaluate(test_data, test_labels)

print('Train Accuracy: %.2f%%'%(train_scores[1]*100))
print('Validation Accuracy: %.2f%%'%(val_scores[1]*100))
print('Test Accuracy: %.2f%%'%(test_scores[1]*100)) 

## CONFUSION MATRIX ##

predic = vg_model.predict(test_data)

predic = np.argmax(predic, axis=1)
labels = np.argmax(test_labels, axis=1)

conf_arr = confusion_matrix(labels, predic)
print(conf_arr)
plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = sn.heatmap(conf_arr, cmap='Blues', annot=True, fmt='d', xticklabels= ['Mild', 'Moderate', 'Normal', 'VeryMild'],
                yticklabels=['Mild', 'Moderate', 'Normal', 'VeryMild'])
plt.title('Alzheimer\'s Diagnosis')
plt.xlabel('Prediction')
plt.ylabel('Truth')
# plt.subplots(figsize=(9, 6))
plt.show(ax)

# get most confused classes
highest_two_missclassified = evaluate_most_confused_labels(conf_arr)
print(highest_two_missclassified)

class_1 = highest_two_missclassified[0][0][0]
class_2 = highest_two_missclassified[0][1][0]


all_possible_classes = ("MildDemented","ModerateDemented","NonDemented","VeryMildDemented")
class_1 = all_possible_classes[class_1]
class_2 = all_possible_classes[class_2]
print(class_1)
print(class_2)

bool_arr = (np.array(all_possible_classes) != class_1) & (np.array(all_possible_classes) != class_2)
output = np.where(bool_arr)
lowest_two_missclassified = np.array(all_possible_classes)[output]
class_3 = lowest_two_missclassified[0]
class_4 = lowest_two_missclassified[1]


# get class_1, class_2 train & test data
easy_train_data_im, easy_train_data_labels, med_train_data_im, med_train_data_labels, hard_train_data_im, hard_train_data_labels = getTieredData(class_1, class_2, train_data_path, 3000, 3100)
easy_test_data_im, easy_test_data_labels, med_test_data_im, med_test_data_labels, hard_test_data_im, hard_test_data_labels = getTieredData(class_1, class_2, test_data_path, 980, 1080)

# get class_3, class_4 train & test data
easy_train_im_class3_class4, easy_train_labels_class3_class4, med_train_im_class3_class4, med_train_labels_class3_class4, unused1, unused2 = getTieredData(class_3, class_4, train_data_path, 100, 200)
easy_test_im_class3_class4, easy_test_labels_class3_class4, med_test_im_class3_class4, med_test_labels_class3_class4, unused1, unused2 = getTieredData(class_3, class_4, test_data_path, 100, 200)


# add other class data to easy, med, hard training and testing sets
nEasy = 100
i = 2
while nEasy > 5:
    nEasy = int(np.floor(easy_train_im_class3_class4.shape[0]/i))
    i = i*2
    print(nEasy)
    
nMed = 100
i = 2
while nMed > 5:
    nMed = int(np.floor(med_train_im_class3_class4.shape[0]/i))
    i = i*2
    print(nMed)

easy_train_data_im = np.append(easy_train_data_im, easy_train_im_class3_class4[0:nEasy], axis=0)
easy_train_data_labels = np.append(easy_train_data_labels, easy_train_labels_class3_class4[0:nEasy], axis=0)
easy_train_data_im = np.append(easy_train_data_im, med_train_im_class3_class4[0:nMed], axis=0)
easy_train_data_labels = np.append(easy_train_data_labels, med_train_labels_class3_class4[0:nMed], axis=0)

med_train_data_im = np.append(med_train_data_im, easy_train_im_class3_class4[nEasy:2*nEasy], axis=0)
med_train_data_labels = np.append(med_train_data_labels, easy_train_labels_class3_class4[nEasy:2*nEasy], axis=0)
med_train_data_im = np.append(med_train_data_im, med_train_im_class3_class4[nMed:2*nMed], axis=0)
med_train_data_labels = np.append(med_train_data_labels, med_train_labels_class3_class4[nMed:2*nMed], axis=0)

hard_train_data_im = np.append(hard_train_data_im, easy_train_im_class3_class4[2*nEasy:3*nEasy], axis=0)
hard_train_data_labels = np.append(hard_train_data_labels, easy_train_labels_class3_class4[2*nEasy:3*nEasy], axis=0)
hard_train_data_im = np.append(hard_train_data_im, med_train_im_class3_class4[2*nMed:3*nMed], axis=0)
hard_train_data_labels = np.append(hard_train_data_labels, med_train_labels_class3_class4[2*nMed:3*nMed], axis=0)

nEasy = 100
i = 2
while nEasy > 5:
    nEasy = int(np.floor(easy_test_im_class3_class4.shape[0]/i))
    i = i*2
    print(nEasy)
    
nMed = 100
i = 2
while nMed > 5:
    nMed = int(np.floor(med_test_im_class3_class4.shape[0]/i))
    i = i*2
    print(nMed)
    
easy_test_data_im = np.append(easy_test_data_im, easy_test_im_class3_class4[0:nEasy], axis=0)
easy_test_data_labels = np.append(easy_test_data_labels, easy_test_labels_class3_class4[0:nEasy], axis=0)
easy_test_data_im = np.append(easy_test_data_im, med_test_im_class3_class4[0:nMed], axis=0)
easy_test_data_labels = np.append(easy_test_data_labels, med_test_labels_class3_class4[0:nMed], axis=0)

med_test_data_im = np.append(med_test_data_im, easy_test_im_class3_class4[nEasy:2*nEasy], axis=0)
med_test_data_labels = np.append(med_test_data_labels, easy_test_labels_class3_class4[nEasy:2*nEasy], axis=0)
med_test_data_im = np.append(med_test_data_im, med_test_im_class3_class4[nMed:2*nMed], axis=0)
med_test_data_labels = np.append(med_test_data_labels, med_test_labels_class3_class4[nMed:2*nMed], axis=0)

hard_test_data_im = np.append(hard_test_data_im, easy_test_im_class3_class4[2*nEasy:3*nEasy], axis=0)
hard_test_data_labels = np.append(hard_test_data_labels, easy_test_labels_class3_class4[2*nEasy:3*nEasy], axis=0)
hard_test_data_im = np.append(hard_test_data_im, med_test_im_class3_class4[2*nMed:3*nMed], axis=0)
hard_test_data_labels = np.append(hard_test_data_labels, med_test_labels_class3_class4[2*nMed:3*nMed], axis=0)


    
##  Class Concentration training and testing

# fixing these labels
easy_train_data_labels = fix_labels(easy_train_data_labels)
#print(easy_train_data_labels)
med_train_data_labels = fix_labels(med_train_data_labels)
#print(med_train_data_labels)
hard_train_data_labels = fix_labels(hard_train_data_labels)
#print(hard_train_data_labels)

easy_test_data_labels = fix_labels(easy_test_data_labels)
#print(easy_test_data_labels)
med_test_data_labels = fix_labels(med_test_data_labels)
#print(med_test_data_labels)
hard_test_data_labels = fix_labels(hard_test_data_labels)
#print(hard_test_data_labels)


# train the model on easy data
init_easy_train = vg_model.evaluate(easy_train_data_im, easy_train_data_labels)
init_easy_test = vg_model.evaluate(easy_test_data_im, easy_test_data_labels)

model_history_easy = vg_model.fit( easy_train_data_im, easy_train_data_labels,
                             validation_data=(easy_test_data_im, easy_test_data_labels),
                             epochs=eps, batch_size=batch, shuffle=True )

train_scores_easy = vg_model.evaluate(easy_train_data_im, easy_train_data_labels)
test_scores_easy = vg_model.evaluate(easy_test_data_im, easy_test_data_labels)

print('Easy Class Concentration Training and Testing:')
print('\n    Before Fit Train Accuracy:       %.2f%%' %(init_easy_train[1]*100))
print('    After Fit Train Accuracy:        %.2f%%' %(train_scores_easy[1]*100))
print('    Overall Baseline Train Accuracy: %.2f%%' %(train_scores[1]*100))
print('\n    Before Fit Test Accuracy:        %.2f%%' %(init_easy_test[1]*100))
print('    After Fit Test Accuracy:         %.2f%%' %(test_scores_easy[1]*100))
print('    Overall Baseline Test Accuracy:  %.2f%%' %(test_scores[1]*100))


# train the model on med data
init_med_train = vg_model.evaluate(med_train_data_im, med_train_data_labels)
init_med_test = vg_model.evaluate(med_test_data_im, med_test_data_labels)

model_history_med = vg_model.fit( med_train_data_im, med_train_data_labels,
                             validation_data=(med_test_data_im, med_test_data_labels),
                             epochs=eps, batch_size=batch, shuffle=True )

train_scores_med = vg_model.evaluate(med_train_data_im, med_train_data_labels)
test_scores_med = vg_model.evaluate(med_test_data_im, med_test_data_labels)

print('Medium Class Concentration Training and Testing:')
print('\n    Before Fit Train Accuracy:       %.2f%%' %(init_med_train[1]*100))
print('    After Fit Train Accuracy:        %.2f%%' %(train_scores_med[1]*100))
print('    Overall Baseline Train Accuracy: %.2f%%' %(train_scores[1]*100))
print('\n    Before Fit Test Accuracy:        %.2f%%' %(init_med_test[1]*100))
print('    After Fit Test Accuracy:         %.2f%%' %(test_scores_med[1]*100))
print('    Overall Baseline Test Accuracy:  %.2f%%' %(test_scores[1]*100))


# train the model on hard data
init_hard_train = vg_model.evaluate(hard_train_data_im, hard_train_data_labels)
init_hard_test = vg_model.evaluate(hard_test_data_im, hard_test_data_labels)

model_history_hard = vg_model.fit( hard_train_data_im, hard_train_data_labels,
                             validation_data=(hard_test_data_im, hard_test_data_labels),
                             epochs=eps, batch_size=batch, shuffle=True )

train_scores_hard = vg_model.evaluate(hard_train_data_im, hard_train_data_labels)
test_scores_hard = vg_model.evaluate(hard_test_data_im, hard_test_data_labels)

print('Hard Class Concentration Training and Testing:')
print('\n    Before Fit Train Accuracy:       %.2f%%' %(init_hard_train[1]*100))
print('    After Fit Train Accuracy:        %.2f%%' %(train_scores_hard[1]*100))
print('    Overall Baseline Train Accuracy: %.2f%%' %(train_scores[1]*100))
print('\n    Before Fit Test Accuracy:        %.2f%%' %(init_hard_test[1]*100))
print('    After Fit Test Accuracy:         %.2f%%' %(test_scores_hard[1]*100))
print('    Overall Baseline Test Accuracy:  %.2f%%' %(test_scores[1]*100))


train_scores_post100 = vg_model.evaluate(train_data_total, train_labels_total)
val_scores = vg_model.evaluate(val_data,val_labels)
test_scores_post100 = vg_model.evaluate(test_data_total, test_labels_total)
print('Train Accuracy (Before Retrain):      %.2f%%'%(train_scores_post100[1]*100))
print('Validation Accuracy (Before Retrain): %.2f%%'%(val_scores[1]*100))
print('Test Accuracy (Before Retrain):       %.2f%%'%(test_scores_post100[1]*100)) 


# retrain total model
model_history_retrain = vg_model.fit( train_data_total, train_labels_total,
                    validation_data=(test_data_total, test_labels_total),
                    epochs=eps, batch_size=batch, shuffle=True )

train_scores_post100 = vg_model.evaluate(train_data_total, train_labels_total)
val_scores = vg_model.evaluate(val_data,val_labels)
test_scores_post100 = vg_model.evaluate(test_data_total, test_labels_total)

predic = vg_model.predict(test_data_total)

predic = np.argmax(predic, axis=1)
labels = np.argmax(test_labels_total, axis=1)

conf_arr = confusion_matrix(labels, predic)
print(conf_arr)
plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = sn.heatmap(conf_arr, cmap='Blues', annot=True, fmt='d', xticklabels= ['Mild', 'Moderate', 'Normal', 'VeryMild'],
                yticklabels=['Mild', 'Moderate', 'Normal', 'VeryMild'])
plt.title('Alzheimer\'s Diagnosis')
plt.xlabel('Prediction')
plt.ylabel('Truth')
# plt.subplots(figsize=(9, 6))
plt.show(ax)

print('Train Accuracy (After Retrain):      %.2f%%'%(train_scores_post100[1]*100))
print('Validation Accuracy (After Retrain): %.2f%%'%(val_scores[1]*100))
print('Test Accuracy (After Retrain):       %.2f%%'%(test_scores_post100[1]*100)) 






















