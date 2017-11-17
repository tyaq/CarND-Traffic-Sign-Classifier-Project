# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[sample-test]: ./examples/test_sample.png "Sample of Test Images"
[class-freq]: ./examples/class_freq.png "Frequency of Sign Types"
[pre]: ./examples/pre_process.png "Image Processing Result"
[ex]: ./examples/ex_signs.png "Traffic Signs to Predict"
[top5]: ./examples/top5_softmax.png "Summery of Results with Top Five Predictions"
[image6]: ./examples/conv1_feature_map_left.png "Feature Map"
[image7]: ./examples/conv2_feature_map_left.png "Feature Map"
[image8]: ./examples/conv1_feature_map.png "Feature Map"
[image9]: ./examples/conv2_feature_map.png "Feature Map"
[image10]: ./examples/conv1_feature_map_st_left.png "Feature Map"
[image11]: ./examples/conv2_feature_map_st_left.png "Feature Map"
[image12]: ./examples/conv1_feature_map_cross.png "Feature Map"
[image13]: ./examples/conv2_feature_map_cross.png "Feature Map"
[image14]: ./examples/conv1_feature_map_int.png "Feature Map"
[image15]: ./examples/conv2_feature_map_int.png "Feature Map"
[image16]: ./examples/conv1_feature_map_speed.png "Feature Map"
[image17]: ./examples/conv2_feature_map_speed.png "Feature Map"

---

### Reflection

### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/tyaq/CarND-TrafficSignClassifier-P2-Habib/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 with 3 color channels (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training set. It is a sample of every sign class.

![alt text][sample-test]

These classes are unevenly distributed throughout the training and validation set. The bar graph below shows each sign classes' frequency in the dataset.

![alt text][class-freq]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

The images are pre-processed by converting them into YUV from RGB using OpenCV. Then the Y-channel(gamma) is extracted, and considered as a grayscale representation of the image. This image is then normalized to occupy the full 0-255 value scale using OpenCV's histogram normalize function. The end result is a grayscale dataset where every image's contrast is the same. Their difference in luminance is the same no image will be too light or too bright.

Here is an example of a traffic sign image before and after this process.

![alt text][pre]

In previous commits, I tried normalizing my the images over the dataset to tone down overexposed, and underexposed pictures. As well as generate new data with random rotations, skew, and scaling to even out the sign classes distribution. However both over these endeavors lead to underfitting of the test dataset. The testing accuracy was significantly lower than the validation accuracy at high epochs. These process where removed from the presented version.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I implemented a LeNet-5 architecture with dropout at the full connection layers. LeNet proved to be good image classification network, avoiding the need to implement ResNet or GoogLeNet for this project. The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gamma image   					    | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten   	      	| outputs 400                				    |
| Fully connected		| outputs 120        							|
| RELU					|												|
| Dropout				|	50% keep probability    					|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Dropout				|	50% keep probability    					|
| Fully connected		| outputs 43        							|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer, a batch size of 128, and 60 epochs. I received a hare better validation accuracy on this run using 60 epochs, but 30 epochs would suffice. The learning rate was kept at 0.001. I found the biggest improvements from fiddling with the hyperparameters was by setting tf.truncated_normals sigma to 0.01. Dropout was set at 50% keep probability.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.984
* validation set accuracy of 0.941 
* test set accuracy of 0.918

These are found in input cell 10 in the IPython notebook.

I opted to use the LeNet architectures from the MNIST lab to bootstrap this project. I first ran it on the traffic signs without tuning the hyperparameters.
This resulted in a validation accuracy around ~89% with test accuracy of 96%. This showed me that LeNet was overfitting the data. So I introduced dropout, initially this was done at every activation, but I found ~91% validation accuracy by only dropping out data in the fully connected layers. I then read [Yann Lechun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) on this problem and noticed he reached better accuracy by converting to grayscale.
Though his neural net architecture was different this transformation of the data should help my neural net focus better on shapes. This provided a nice incremental improvement to ~94% validation accuracy, with ~98% testing acccuracy. The gap was only 4% in the overftting problem.
Since I was already working on the image processing aspect I decided now would be great opportunity to even out my classes frequency graph. Having some prominent classes was skewing the model towards predicting them more often.So I generated new data, by randomly augmenting images in the lower frequency classes. The augmentations included rotating, scaling, and skewing the images. This however led to underfitting and I'm still uncertain as to why. I then shifted my focus to tuning the hyperparameters, this was done by increasing and decreasing the parameters until they provided better validation accuracy. 
Most of them remained unchanged, but I saw significant gains, by setting the sigma used in tf.truncated_normals to 0.01. At this point I realized that I should change my architecture to get better results. However faced with having to change the architecture and rerunning these experiments, I decided to put an end to this tweaking. 

I decided my model was good enough. The model definitely overfits, but a with a test accuracy of 92% it is accurate enough to identify my supplied signs.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found using Google StreetView in Nuremberg, Germany:

![alt text][ex]

The first three images(Turn Left,Stop, Go Straight or Right) should all be pretty easy to classify because they are taken face on and in the day time.

However the third image(No Parking on Right) should be difficult to classify because this sign is no where in our data set. I threw it in because it is a real sign in Germany and I wanted to see if the net prioritizes the arrow, cross, or round shape.

Also, the fourth image(Right of Way Ahead) might be difficult because the picture was taken in a shadow. However our preprocessing should solve that problem.

The sixth image(Speed Limit 30km/h) might be difficult because in this city speed limit signs are posted on rectangles, not circles like in our data set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

![alt text][top5]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn Left Ahead  		| Turn Left Ahead   							| 
| Stop    		       	| Stop 	    									|
| Go Straight or Right 	| Go Straight or Right   						|
| No Parking on Right	| Keep Right									|
| Right of Way Ahead	| Right of Way Ahead			 				|
| Speed Limit 30km/h	| Priority Road      							|


The model was able to correctly guess 4 of the 5 traffic signs in the test set, which gives an accuracy of 80%. This compares worse to the accuracy on the test set of 92%. However the speed limit sign is kind of a gotcha question, because it is a rectangle while the training was on round. I am surprised that it predicts priority road, a diagonal sign, with 90% confidence. The correct answer only has 1% confidence.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 12th cell of the IPython notebook.

For every image, the model has high confidence in its predictions. The top 5 probabilities are best seen above.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1               		| Turn Left Ahead   							| 
| 1        		       	| Stop 	    									|
| 0.95               	| Go Straight or Right   						|
| 0.90                 	| Keep Right									|
| 0.95              	| Right of Way Ahead			 				|
| 0.94              	| Priority Road      							|

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image6]
![alt text][image7]

We can see the CNN prioritizing the round shape, with arrow in th middle.

![alt text][image8]
![alt text][image9]

We can see the CNN prioritizing the hexagon shape, with stop lettering in th middle.

![alt text][image10]
![alt text][image11]

We can see the CNN prioritizing the round shape, with straight arrow in th middle.

![alt text][image12]
![alt text][image13]

We can see the CNN prioritizing the arrow in the sign and it picks up some roundness.

![alt text][image14]
![alt text][image15]

We can see the CNN prioritizing the triangle shape, and some random pixels in the middle.

![alt text][image16]
![alt text][image17]

We can see the CNN prioritizing identifying the round shape, but also the rectangles of the sign. In the second convolution the round shape looks more like the edges of the diamond.
