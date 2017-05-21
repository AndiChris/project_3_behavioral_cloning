# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_2017_05_13_16_37_38_669.jpg "Center Driving"
[image2]: ./images/center_2017_05_14_16_24_29_041.jpg "Recover Driving 1"
[image3]: ./images/center_2017_05_14_16_24_29_736.jpg "Recover Driving 2"
[image4]: ./images/center_2017_05_14_16_24_30_187.jpg "Recover Driving 3"
[image5]: ./images/flipped.jpg "Flipped Image"
[image6]: ./images/metrics.png "Metrics"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 70-80) 

The model includes RELU layers to introduce nonlinearity (code line 73 & 75), and the data is normalized in the model using a Keras lambda layer (code line 71). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 15). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also included 2 laps counter-clockwise so the model can generalize better. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the well known LeNet architecture. I thought this model might be appropriate because it has proven to be a good starting point in the last traffic-sign-classifier problem.

In order to gauge how well the model was working I trained the model and just run the simulator and observed its behavior.

Surprisingly up to the curve after the bridge the model was doing really well from the beginning. Though adding additional training data (i.e. adding data for this problematic curve) made the performance worse. I think I just had hit a sweet spot with good and probably balanced training data in the first place.

However to get the vehicle drive a full track autonomously I had to enhance my preprocessing steps and augmented my data a little bit better.

#### 2. Final Model Architecture

The final model architecture (model.py lines 70-80) consisted of a convolution neural network with the following layers and layer sizes:

1) Pre-processing layer to normalize pixel data around 0. Input shape is the actual image shape. 160x320px with 3 color channels.
2) Pre-processing layer to cut-out everything but the road which is of interest. Cut 25px from bottom and 65 from the top.
3) First layer of the CNN. A convolutional layer with filter-depth of 6 and filter-width/height of 5x5. The stride parameter was set to 5 and  for the activation a Relu-function was choosen. 
4) MaxPooling layer with default parameters of 2x2 pooling-size, 2 as stride and valid padding.
5) Same as 3.
6) Same as 4.
7) Flatten Output of 6.
8) Fully connected layer with output 120
9) Fully connected layer with output 84
10) Fully connected layer with output 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center when it is a little off. These images show what a recovery looks like from left side to center:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on the same track but driving counter-clockwise in order to get more data points.

I also used the images from the left and right camera with an +,- 0.25 steering correction to get even more data.

To augment the data sat, I flipped images and angles thinking that this would help generalizing the model handling left and right turns equally good. For example, here is an image that has then been flipped:

![alt text][image5]

As a last step all images are converted to HSV color space. I hoped that this would make the model perform better in general and especially at the curve just before the dirt track. There the right road-border has a completely different color. So instead letting the model learn with hue (H) saturation (S) and value (V) should be better.

After the collection process, I had 32100 number of data points.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. A good number of epochs was 4 as indicated by a not much more reduction of loss in the last epochs.

![alt text][image6]

Because I used an adam optimizer manually adjusting the learning rate wasn't necessary.
