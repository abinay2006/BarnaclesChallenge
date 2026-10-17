## DALI Barnacles Challenge

Initial Brainstorming:

The main problem presented by the scientist is the high effort and time it takes to manually count the barnacles in an image. Thus, we can automate this process using image analysis tools, such as OpenCV and machine learning algorithms. 

- Use OpenCV to break down the image by first cropping to the desired portion (inside the green border), 
- Determining the contours presented in the image 
- Develop a characteristic dataset with features like the shape, color, and edges
- Use the masked image to generate labels - determine if a contour is actually a barnacle or not depending on whether the masks has a barnacle there or not
- Once have the x (features) and y (labels) in the dataset has been created, I will implement Random Forest to train the model to determine whether something is a barnacle or not using the two datapoint (image 1 and image 2) 
- Test the ML model on the unseen images to determine the number of barnacles present


Subtasks
Subtask
Descrition
Input
Output
Image processing
Use OpenCV’s ROI (region of interest) functioanlity to process the image and the mask and crop the region to inside the green bars (if present) 
Field images taken by the scientists 
The cropped image, containing only the region inside the green border to count
Contour Detection
Use OpenCV functionality like contour analysis to detect these regions using the edges found from the Canny function in the grayscale image
The cropped, grayscale image of barnacles with edges detected
Highlighted image showing contours in the input image
Extract features (x)
From all the contours detected, extract features like its shape, color gradient, and circularity to distinguish barnacles from noise (e.g., background)
The image with all the contours detected
A dataset containing values for the different features of each detected contour
Generate the labels (y)
From the provided masked image, determine if the contours found are barnacles or not, depending on whether they overlap with the highlighted barnacles in the masked image
- Image showing the contours 
- the masked image
Binary label for each contour: barnacle or not (1 or 0)
Train Machine Learning algorithm
Use the Random Forest machine learning algorithm (supervised) to train to identify barnacles given the dataset with the features and the labels
- Data of barnacle features - the binary label dataset
Trained Model
Assess Model Performance
Train the machine learning model on the dataset for image1 provided and then assess its performance using image 2 since its label can also be created and analyzed
- Dataset of features and labels of image 1 to train
- Dataset of features and labels of image 2 to test
Model accuracy, precision, recall, and F1 score
Counting and review
Run the machine learning algorithm on the contours of the unseen images to count the number of barnacles present and help the scientists!
New unseen images
A predicted count value for barnacles in those images



Specific subtask and its performance

A key step in building this model is the machine learning classification model that will be used to classify future unseen images and determine the number of barnacles. The ML model will use the dataset of features and labels created by img1 and img 2 to train on whether something is a barnacle or not and use it to count barnacles in unseen images in later stages.

To evaluate the model's performance, I can use the classification_report function in the Python scikit-learn library. This gives important metrics such as accuracy, precision, recall, and F1 score. 

The accuracy would help me evaluate how well the model was trained and is performing on unseen data to gauge its usefulness. Precision will help determine the trustworthiness of our "yes" labels (whether the contours predicted to be barnacles are actually barnacles or whether the model incorrectly judges other objects as barnacles too).
Recall will help in determining how well the model finds all the correct "yes" cases (barnacles) present in the data. F1 score is a good metric because it balances recall and precision, since it is the harmonic mean of the two values. These would help guide parameter tuning depending on which value is low/high.

After running the model on unseen data, I can manually compare how many barnacles are highlighted with the actual number present in the image and use metrics (precision, recall, and F1 score) to assess overall performance.

Since we were only provided with two masked images, the machine learning algorithm has very limited data, so it will not perform that well. We can still calculate precision, recall, and F1 by training on img 1 and testing on img 2 because we can create output labels for both. However, this is very limited data, and having more would improve accuracy and lead to better predictions. 


Prototype:

[insert github link]

Prototype Analysis:
	
After tuning a few parameters and testing a few ML algorithms, I chose Random Forest
as it performed the best. The detection and performance calculated based on image 2 is shown below.

   Performance on Image 2:

 Label detection based on mask		  Performance of ML model


    Evaluation on image 2:


Focusing on the row with Barnacle detection performance, we can see that the model’s accuracy is not that high at 60%. Looking at the precision and recall, we can see that the recall is high, meaning it is able to detect most of the barnacles. However, the precision is low, meaning it also detects things that are not barnacles (e.g., background) as barnacles. 

We can also visually see that the ML algorithm performed worse at identifying abrnacles than the actual labeled dataset. This is probably because we trained the model on only one data point, image 1. But if we add more data, accuracy will also increase. 

We also have other unseen images that we can use the model to predict the number of barnacles in. However, we can't evaluate performance because we don’t have the labels (or masks) for them. Based on training on two data points (img1 and img2), my model predicted 821 barnacles for unseen image 1 and 96 barnacles for unseen image 2.
	



Conclusions:
Barnacles have specific features that can be used as datapoints for its identification, including its circularity, color, brightness, and number of edges. 
If greater number of images require barnacle detection, automation of a highly accurate model would be needed.
A few issues with my model making it hard to pursue
Accuracy too low (probably can be increased if more training data and label is present)
Requires cropping to the desired region manually for each image
Possible solution: cropping to the desired region manually for each image is not realistic if the number of data points increase to thousands or millions.
Would need automatic detection of the cropped region for all training data by looking at the already cropped mask given. Then only for additional unseen (test) images, crop would be required to the desired area.
The greater the number of data, the higher the accuracy!
Test with tuning parameters on the machine learning algorithms to improve the accuracy further or try contemporary algoriuthms (e.g., BERT, ELM).

My current approach works for small amount of data where a human can look at all the images and crop to the area of interest. Additionally, since we can consider more factors that would be unique to a barnacle to improve the accuracy further such as texture or standard deviation of the colors and brightness. 
Human feedback on the model would be crucial because it would help refine the model further as it learns from it. 
For example, once the model predicts possible barnacle candidate, a human can look at and reject possible candidates or add new ones. This is especially important for barnacles in a close proximity that seem to be hierarchical and harder to detect.
For images with bad lightning or blurry images, a human can better identify missed barnacles
Additionally, if specific information regarding the measurement of the barnacles are known or their clustering behavior, those could be added as additional factors for the model to consider and run trial on.



Tell us about your learning process. As stated, one of the main points of this application process is for you to learn new things. What is something you learned that excited you?

I greatly enjoyed working on this project and learned a lot of new things in the process. I was particularly excited by the opportunity to understand a lot of OpenCv functionalities and see the way they work. For example, in the past, I had not used the selectROI function, but using it for this project made me realize its uses and importance that I will implement in my future projects. 

Additionally, I greatly enjoyed learning about implementing machine learning from the skit-learn in Python. Previously, my experience with ML algoiithms were mostly constricted to the R programming language. But through this project, I realized how accesible and fun to use the Python libraries and how they can be used to solve even further complex algorithms due to the wide array of features available in libraries like scikit-learn.

Lastly, I enjoyed the problem solving and debugging aspect of this project. I stated with a strong understanding of the problem and then laid out each step and the particular task so I had a deep udensrtanding of how the different image analysis and machine learning parts were working together. Additionally, in prior projects, I would just find performance metrics, like accuracy, recall, and precision and not think about the values but more focused on just improving it. However, through this project, I spent time actually understand what each value meant in this perspective and what a low vs high value was telling me, which helped in tuning some parameters like overlay.

I found this project to be very rewarding in terms of learning new tools and apply them to a real-world problem.
