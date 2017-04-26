#Tips
#Extract full HOG features just once for entire ROI in each frame

#Steps of the project
# - Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# - Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
# - Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# - Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# - Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# - Estimate a bounding box for vehicles detected.

# Pipeline
# 1. Pre-process data
# 2. Extract features
# 3. Classify using model (SVM/DT/Neural network)

import time
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label
from sklearn import svm
from sklearn import grid_search
from moviepy.editor import VideoFileClip

class VehicleDetect():
    def __init__(self):
        self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 16  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32) # Spatial binning dimensions
        self.hist_bins = 64    # Number of histogram bins
        self.spatial_feat = False # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.y_start_stop = [400, 656] # Min and max in y to search in slide_window()
        self.X_scaler = None
        self.clf = None
        self.hist_range = (0, 256)
        self.debug = False
        self.bboxes = []

params = VehicleDetect()

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def convert_color(img, conv):
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    return np.copy(img)

def single_img_features(img, params):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, params.color_space)
    #3) Compute spatial features if flag is set
    if params.spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=params.spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if params.hist_feat == True:
        hist_features = color_hist(feature_image, nbins=params.hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if params.hog_feat == True:
        if params.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    params.orient, params.pix_per_cell, params.cell_per_block, 
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,params.hog_channel], params.orient, 
                        params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def extract_features(imgs, params):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for fname in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(fname)
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, params.color_space)

        if params.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=params.spatial_size)
            file_features.append(spatial_features)
        if params.hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=params.hist_bins, bins_range=params.hist_range)
            file_features.append(hist_features)
        if params.hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if params.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        params.orient, params.pix_per_cell, params.cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,params.hog_channel], params.orient, 
                            params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=True)

            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(img, windows, params):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, params)
        #5) Scale extracted features to be fed to classifier
        test_features = params.X_scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = params.clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def find_cars(img, ystart, ystop, scale, params, index=0):

    #draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,640:1280,:]
    ctrans_tosearch = convert_color(img_tosearch, params.color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // params.pix_per_cell)-1
    nyblocks = (ch1.shape[0] // params.pix_per_cell)-1 
    nfeat_per_block = params.orient*params.cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // params.pix_per_cell)-1 
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, params.orient, params.pix_per_cell, params.cell_per_block, feature_vec=False, vis=False)
    hog2 = get_hog_features(ch2, params.orient, params.pix_per_cell, params.cell_per_block, feature_vec=False, vis=False)
    hog3 = get_hog_features(ch3, params.orient, params.pix_per_cell, params.cell_per_block, feature_vec=False, vis=False)

    if params.debug == True:
        print("Debug is on")
        himg11 = np.dstack((himg1, himg2, himg3))*255
        fig5 = plt.figure()
        plt.imshow(himg11)
        plt.title('HOG visualization')
        fig5.savefig("output_images/hog_visualize_{0}.png".format(index))

    bbox_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if params.hog_channel == "ALL":
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            if params.hog_channel == 0:
                hog_features = hog_feat1
            if params.hog_channel == 1:
                hog_features = hog_feat2
            if params.hog_channel == 2:
                hog_features = hog_feat3

            xleft = xpos*params.pix_per_cell
            ytop = ypos*params.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            if params.spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=params.spatial_size)
            else:
                spatial_features = []
            if params.hist_feat == True:
                hist_features = color_hist(subimg, nbins=params.hist_bins)
            else:
                hist_features = []

            # Scale features and make a prediction
            test_features = params.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = params.X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = params.clf.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale) + 640
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    #return draw_img, bbox_list
    return img, bbox_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def process_video(image):
    global params

    params.debug = False
    ystart = 400
    ystop = 656
    scale = 1.5

    out_img, hot_windows = find_cars(image, ystart, ystop, scale, params)
    params.bboxes.append(hot_windows)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    params.bboxes = params.bboxes[-10:]

    #Add heat to each box in box list
    heat = add_heat(heat, [bbox for sublist in params.bboxes for bbox in sublist])

    #Apply threshold to help remove false positives
    heat = apply_threshold(heat, 5)

    #Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    #Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img

def search_cars(fname, params, index):
    image = mpimg.imread(fname)
    draw_image = np.copy(image)
    test_image = np.copy(image)

    image = image.astype(np.float32)/255

    ystart = 400
    ystop = 656
    scale = 1.5

    hog_image = mpimg.imread(fname)

    #where is box list from this function?
    print("Find cars using HoG sub-sampling")
    out_img, hot_windows = find_cars(hog_image, ystart, ystop, scale, params, index)

    test_image = draw_boxes(test_image, hot_windows)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    print("Add heat to each window")
    #Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    
    #Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    #Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    #Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_image = draw_labeled_bboxes(draw_image, labels)

    return heatmap, test_image, labels, draw_image

#Main function start

plt.close('all')

print("Start vehicle detection")

cars = []
notcars = []

print("Read training file names")

images = glob.glob('vehicles/GTI_Far/*.png')
for index, im in enumerate(images):
    cars.append(im)
images = glob.glob('vehicles/GTI_Left/*.png')
for im in images:
    cars.append(im)
images = glob.glob('vehicles/GTI_MiddleClose/*.png')
for im in images:
    cars.append(im)
images = glob.glob('vehicles/GTI_Right/*.png')
for im in images:
    cars.append(im)
images = glob.glob('vehicles/KITTI_extracted/*.png')
for index, im in enumerate(images):
    if index < 1500:
        cars.append(im)

images = glob.glob('non-vehicles/GTI/*.png')
for index, im in enumerate(images):
    notcars.append(im)

car_features = []
notcar_features = []

print("Number of cars " + str(len(cars)))
print("Number of notcars " + str(len(notcars)))

print("Extract features")

car_features = extract_features(cars, params)
notcar_features = extract_features(notcars, params)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
params.X_scaler = X_scaler
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define a labels vector based on features lists
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

car_img = mpimg.imread(cars[car_ind])
notcar_img = mpimg.imread(notcars[notcar_ind])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.tight_layout()
ax1.imshow(car_img)
ax1.set_title('Car')
ax2.imshow(notcar_img)
ax2.set_title('Not car')
f.savefig('output_images/car_notcars.png')

# Plot an example of raw and scaled features
f1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,4))
f1.tight_layout()
temp, car_hog_img = get_hog_features(car_img[:,:,0],
                               params.orient, params.pix_per_cell, params.cell_per_block,
                               vis=True, feature_vec=True)
ax1.imshow(car_img[:,:,0], cmap='gray')
ax1.set_title('Car CH-1')
ax2.imshow(car_hog_img, cmap='gray')
ax2.set_title('Car CH-1 Hog')
temp, notcar_hog_img = get_hog_features(notcar_img[:,:,0],
                               params.orient, params.pix_per_cell, params.cell_per_block,
                               vis=True, feature_vec=True)
ax3.imshow(notcar_img[:,:,0], cmap='gray')
ax3.set_title('not-Car CH-1')
ax4.imshow(notcar_hog_img, cmap='gray')
ax4.set_title('not-Car CH-1 Hog')
f1.savefig('output_images/HOG_example.jpg')

# Split up data into randomized training and test sets
print("Split training data")
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

#SVM classifier
# Parameters: Kernel, C, gamma
# Linear kernel : C
# Non-linear kernel : C and gamma
# Algorithms for automatic parameter search
# - GridSearchCV
# - RandomizedSearchCV

#Using GridSearchCV
print("Grid search SVM params")
parameters = {'kernel':('linear',), 'C':[0.1,0.5,0.8,1,3,7,10]}
svr = svm.SVC(cache_size=10000)
clf = grid_search.GridSearchCV(svr, parameters)
params.clf = clf
#access tuned parameters using clf.best_params_
print("Start training")

# Check the training time for the SVC
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#Detect vehicles in test images
images = glob.glob('test_images/*.jpg')
for index, im in enumerate(images):
    heatmap, bboxes, labels, draw_img = search_cars(im, params, index)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
    ax1.imshow(bboxes)
    ax1.set_title('Windows')
    ax2.imshow(heatmap, cmap='hot')
    ax2.set_title('Heat Map')
    ax3.imshow(draw_img)
    ax3.set_title('Boxes')
    fig.tight_layout()
    fig.savefig('output_images/bboxes_and_heat{0}.png'.format(index))

    fig, (ax1) = plt.subplots(1, 1, figsize=(12,4))
    ax1.imshow(draw_img)
    ax1.set_title('Output')
    fig.savefig('output_images/output_bboxes{0}.png'.format(index))

plt.close('all')

project_output = 'project_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(process_video)
project_clip.write_videofile(project_output, audio=False)
