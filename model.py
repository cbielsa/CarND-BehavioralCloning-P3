
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import cv2
import os.path



# 1. FUNCTION DEFINITION ============================================

# Crop given number of top and bottom rows from image
#   rationale:
#     - Bottom rows contain part of the car body, with whose presence
#       we could not simulate car shifts across the road by horizontally
#       shifting the image
#     - Top rows contain landscape features unrelated to the road that
#       we want to prevent the model from memorizing
#     - The fewer the number of pixels, the faster model training
#       and inference

def crop_top_bottom_rows(img, numRowsBottomCropped=20, numRowsTopCropped=55):
    imgMod = np.empty((img.shape[0]-numRowsBottomCropped-numRowsTopCropped, img.shape[1], img.shape[2]), dtype=img.dtype)
    imgMod[:] = img[numRowsTopCropped:img.shape[0]-numRowsBottomCropped]
    return imgMod


# Image pre-processing, performed on all original images both for training and for inference.
# Pre-processing consists of:
#   - cropping of a number of top and bottom rows
#   - image resizing to 32x120 pixels (to reduce training and inference time)
#   - data-type conversion to float32
#   (normalization is done later on with a lambda Keras layer at the beginning of the CNN)

def preprocessImage(img, numRowsBottomCropped=20, numRowsTopCropped=55, newNumRows=32, newNumCols=120):
    img = crop_top_bottom_rows(img, numRowsBottomCropped, numRowsTopCropped)
    return cv2.resize(img, (newNumCols, newNumRows)).astype(np.float32)  # retype and resize


# Randomly transform image color channels.
# First, image is transform to HSV color space.
# Then, random disturbances within given input ranges are applied to each channel.

def transform_colorChannels(img, minH=0.8, maxH=1.2, minS=0.7, maxS=1, minV=0.40, maxV=1.2):
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    random_H = minH + np.random.uniform()*(maxH-minH)
    random_S = minS + np.random.uniform()*(maxS-minS)
    random_V = minV + np.random.uniform()*(maxV-minV)
    img[:,:,0] = img[:,:,0]*random_H
    img[:,:,1] = img[:,:,1]*random_S
    img[:,:,2] = img[:,:,2]*random_V
    
    img = np.minimum(img, 255)
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    
    return img



# Randomly shift image both vertically and horizontally.
#   - Vertical shifts simulate road slope.
#   - Horizontal shifts simulate shifts in car position across the road.
#     For each pixel of the horizontal shift, steering angle is modified by 0.01 rad (0.57 deg)

def transform_shift(img, steer, hShiftMin=-35, hShiftMax=35, vShiftMin=-10, vShiftMax=10 ):
    
    random_hShift = hShiftMin + np.random.uniform()*(hShiftMax-hShiftMin)
    random_vShift = vShiftMin + np.random.uniform()*(vShiftMax-vShiftMin)
    
    shiftMatrix = np.float32( [[1,0,random_hShift], [0,1,random_vShift]] )
    imgMod = cv2.warpAffine(img, shiftMatrix, (img.shape[1],img.shape[0]))
    steerMod = steer + 0.01*random_hShift  # 0.01 rad (0.57 deg) per pixel
    return imgMod,steerMod



# Add random shadowed region to image.
#   - For each image, a shadow is introduced with a probability pShadow
#   - If a shadow is introduced, then two random image points are chosen,
#     a line is formed by connecting the two points, and the region to one
#     of the sides of the line is darkened by multiplying the L-channel
#     of the HLS-version of the image by a random number in the range 0.6-1.0.
#   - Image is transformed to HLS rather than HSV --as done in transform_colorChannels,
#     to add some more variety to the augmentation process.
#
#  Acknowledgement: this function is a modified version of the function shared by Vivek Yadav
#  in https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.cyehlucpj

def transform_shadow(img, pShadow=0.5, brightMin=0.6, brightMax=1.0):
       
    # transform image to HLS
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    x1 = np.random.uniform()*img.shape[1]
    x2 = np.random.uniform()*img.shape[1]
    y1 = np.random.uniform()*img.shape[0]
    y2 = np.random.uniform()*img.shape[0]
    
    X_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]
    
    shadow_mask = 0*img[:,:,1]
    shadow_mask[( (X_m-x1)*(y2-y1) - (x2-x1)*(Y_m-y1) >=0 )]=1
    
    if np.random.uniform() < pShadow:
        
        brightness =  brightMin + np.random.uniform()*(brightMax-brightMin)
        cond0 = shadow_mask==0
        cond1 = shadow_mask==1
        
        # randomly choose which side to shade
        if np.random.randint(2)==1:
            img[:,:,1][cond1] = img[:,:,1][cond1]*brightness
        else:
            img[:,:,1][cond0] = img[:,:,1][cond0]*brightness    
    
    # transform back to RGB
    return cv2.cvtColor(img, cv2.COLOR_HLS2RGB)



# Vertically flip image with a probability pFlip, to balance left and right curves
# in training dataset.
# If the image is flipped, the sign of the steering angle is inverted.

def transform_flip(img, steer, pFlip=0.5 ):
    if np.random.uniform() < pFlip:
        img = cv2.flip(img, 1)
        steer = -steer
    return img, steer



# Apply random horizontal shearing to image, to simulate shift in car orientation.
# Pixels in lowest row remain fixed, while higher pixel rows are shifted
# horizontally in proportion to their vertical distance.

# Acknowledgement: this function is a modified version of the function shared
# by Kaspar Sakmann in https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.lvn2kq3ze

def transform_shear(img, steer, shearPixelsRange=30):
    
    rows, cols, ch = img.shape
    
    dx = np.random.randint(-shearPixelsRange, shearPixelsRange+1)
    
    points1 = np.float32([[0,rows], [cols,rows], [cols/2,rows/2]])
    points2 = np.float32([[0,rows], [cols,rows], [cols/2+dx, rows/2]])
    
    M = cv2.getAffineTransform(points1, points2)
    img = cv2.warpAffine(img, M, (cols,rows), borderMode=1)
    steer += 0.010*dx  # 0.01 rad (0.57 deg) per pixel
    
    return img, steer


# This is the function used for data augmentation
# It applies a number of random transformations to the original image,
# including:
#   - vertical flip, to balance right and left curves in dataset.
#   - color channel transformation.
#   - addition of random shadows.
#   - horizontal shear transformation, to simulate car rotation.
#   - horizontal shifts, to simulate car translations across the road.
#   - vertical shifts, to simulate road slope.

def random_transform(img, steer):
    img, steer = transform_flip(img, steer)
    img = transform_colorChannels(img)
    img = transform_shadow(img)
    img, steer = transform_shear(img, steer)
    img, steer = transform_shift(img, steer)
    
    return img, steer



# Python generator of augmented train data for fit_generator.
# It draws random sample batches from the input dataset and applies random transformations.

def generate_train_data(X, y, batch_size=32):
    
    batch_X = np.empty((batch_size, *X_train.shape[1:]))
    batch_y = np.empty(batch_size)

    while 1:

        # draw random sample from original train dataset
        index = np.random.choice(len(X), batch_size)
        
        # randomly transform images and adjust steering angle accordingly
        for i_batch in range(batch_size):
            batch_X[i_batch], batch_y[i_batch] = random_transform( X[index[i_batch]], y[index[i_batch]] )

        yield batch_X, batch_y




# 1. DATA PRE-PROCESSIGN ==========================================

print('\nLoading, pre-processing and splitting dataset...\n')

# paths containing training data
datapath = 'drive_circ1_merged/'
sideImages_datapath = 'drive_circ1_sideImages/'

# parse image filenames and steering angles from CSV files
# for images taken with car side cameras, we apply a 0.1 rad offset
# to the steering angle at the time the image was taken
# (1m side camera offset/10m until shift is corrected ~0.1 rad)

imgfileList = []
steerList = []

with open(datapath+'driving_log.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        imgfileList.append(row[0].replace('\\','/').lstrip())  # windows compatibility
        steerList.append(row[3])

iNumCenterImg = len(imgfileList)
        
deltaSteerSideImages = 0.1  # arctan( 1 m camera offset / 10m until shift is corrected )

with open(sideImages_datapath+'driving_log.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        # left camera image
        imgfileList.append(row[1].replace('\\','/').lstrip())  # windows compatibility
        steerList.append( float(row[3])+deltaSteerSideImages )
        # right camera image
        imgfileList.append(row[2].replace('\\','/').lstrip())  # windows compatibility
        steerList.append( float(row[3])-deltaSteerSideImages )        

# pre-process labels
y = np.array(steerList, np.float32)

# shape of augmented images fed to the model
imgshape = (32, 120, 3) 


# Load pre-processed images to memory

X = np.empty( (len(imgfileList), *imgshape), np.float32)

# Load and pre-process images from centre camera
for i in range(iNumCenterImg):
    fname = datapath+imgfileList[i]
    if os.path.isfile(fname): 
        X[i] = preprocessImage( mpimg.imread(fname) )
    else:
        print('image file '+fname+' does not exist!')

# Load and pre-process images from side cameras
for i in range(iNumCenterImg, len(imgfileList)):
    fname = sideImages_datapath+imgfileList[i]
    if os.path.isfile(fname): 
        X[i] = preprocessImage( mpimg.imread(fname) )
    else:
        print('image file '+fname+' does not exist!')


# Split data in 80% training and 20% validation sets

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=785949)

del X; del y



# 2. MODEL ARCHITECTURE ==========================================

print('\nConstructing and training Convolutional Neural Network...\n')

from keras.models import Sequential
from keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Lambda
from keras.optimizers import Adam

def model_new():
    
    model = Sequential()
    
    model.add( Lambda(lambda x: x/255-0.5, input_shape=imgshape) )
    
    model.add( Convolution2D(12, 4, 4, border_mode='valid', init='normal') )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )
    
    model.add( Convolution2D(18, 4, 4, border_mode='valid', init='normal') )
    model.add( Activation('relu') )
    
    model.add( SpatialDropout2D(0.2, dim_ordering='tf') )
    model.add( Convolution2D(24, 3, 3, border_mode='valid', init='normal') )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )
    
    model.add( Dropout(0.3) )
    model.add( Convolution2D(32, 3, 3, border_mode='valid', init='normal') )
    model.add( Activation('relu') )
       
    model.add( Dropout(0.5) )
    model.add( Flatten() )
    model.add( Dense(32, activation='relu', init='normal') )

    model.add( Dense(12, activation='relu', init='normal') )
    
    model.add( Dense(1, init='normal') )
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    
    return model


print()
print(model_new().summary())



# 3. TRAIN MODEL ==========================================

# hypterparameters
N_EPOCHS = 20
BATCH_SIZE = 64
SAMPLES_PER_EPOCH = 10000

# create model
model = model_new()

# train model with batches of augmented data
history = model.fit_generator(
    generate_train_data(X_train, y_train, BATCH_SIZE),
    samples_per_epoch=SAMPLES_PER_EPOCH,
    nb_epoch=N_EPOCHS,
    validation_data=(X_val, y_val) )

print("\nTraining Mean Absolute Error :", history.history['mean_absolute_error'][-1])
print("Validation Mean Absolute Error :", history.history['val_mean_absolute_error'][-1], '\n')



# 4. SAVE MODEL ARCHITECTURE AN WEIGHTS ====================

print('\nSaving model...\n')

# save model architecture
with open("model.json", 'w') as file:
    file.write( model.to_json() )

# save model weights
model.save_weights("model.h5")


