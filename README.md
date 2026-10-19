# DALI Barnacles Challenge
This project uses computer vision and machine learning approach to automate barnacle image analysis. It uses image processing techniques to identify barnacle regions and extract features, then applies the Random Forest machine learning classifier to distinguish the barnacle from the surroundings. 

# Files 
- Images: All the images given of the barnacles and the masks
- README: Project overview, setup instructions, and project documentation.
- trial.py: Main script that runs the image processing, feature extraction, classification, and visualization.
  

# Requirements 
- Python
- OpenCV (opencv-python)
- NumPy
- scikit-learn
- Matplotlib

This can be done through the following line in the terminal:
pip install opencv-python numpy scikit-learn matplotlib

# How to run 
1. Clone or download the repository.
2. Install the dependencies listed above.
3. Make sure that the images are available at the path specified in trial.py.
4. Run on terminal using this line: "python trial.py"
5. When an image window appears, select the region of interest by dragging a blue square, then press Enter to continue. Repeat this process for all images given.

The trial.py file contains all the code to run this prototype. The output of this file provides the barnacle detection and accuracy for img2 as well as the predicted barnacle count and possible candidate barnacle highlighted for the two unseen images given (unseen_img1 and unseen_img2). 

# 1. Initial Brainstorming

The main problem presented by the scientist is the high effort and time it takes to manually count the barnacles in an image. Thus, we can automate this process using image analysis tools, such as OpenCV and machine learning algorithms. 

I will develop a semi-autonomous system to detect the barnacles. To do so, I will first use OpenCV to break down the image and crop to the desired portion (inside the green border). Then I will determine the different shapes present in the image and look at each of them for different characteristics like the shape, color, and texture. Then, using the masked image provided, I will generate the output label indicating whether a particular shape is a barnacle or not. This will be used to train the supervised Machine Learning algorithm. Finally, I will test the ML model on the image provided to see its performance and then also test on the unseen images to determine the number of barnacles present there.

# 2. Subtasks

<img width="597" height="668" alt="Screenshot 2026-10-18 at 8 24 47 PM" src="https://github.com/user-attachments/assets/88aa244d-fa2d-4c81-8705-4f71769852a8" />

Below are the specific characteristics of a barnacle used to create the training dataset, along with their average values.

<img width="583" height="635" alt="Screenshot 2026-10-18 at 8 25 57 PM" src="https://github.com/user-attachments/assets/d53ec33d-9edc-4d31-bce3-3bc6448a7dc7" />

# 3. Specific subtask and its performance 

A key step in building this model is the machine learning classification model that will be used to classify future unseen images and determine the number of barnacles. The ML model will use the dataset of features and labels created by img1 and img 2 to train on whether something is a barnacle or not and use it to count barnacles in unseen images in later stages.

To evaluate the model's performance, I can use the classification_report function in the Python scikit-learn library. This gives important metrics such as accuracy, precision, recall, and F1 score. 

The accuracy would help me evaluate how well the model was trained and is performing on unseen data to gauge its usefulness. Precision will help determine the trustworthiness of our "yes" labels (whether the contours predicted to be barnacles are actually barnacles or whether the model incorrectly judges other objects as barnacles too).
Recall will help in determining how well the model finds all the correct "yes" cases (barnacles) present in the data. F1 score is a good metric because it balances recall and precision, since it is the harmonic mean of the two values. These would help guide parameter tuning depending on which value is low/high.

After running the model on unseen data, I can compare how many barnacles are highlighted with the actual number present in the image and use metrics (precision, recall, and F1 score) to assess overall performance.

Since we were only provided with two masked images, the machine learning algorithm has very limited data, so it will not perform that well. We can still calculate precision, recall, and F1 by training on img 1 and testing on img 2 because we can create output labels for both. However, this is very limited data, and having more would improve accuracy and lead to better predictions.

# 4. Prototype Analysis

After tuning a few parameters and testing a few ML algorithms, I chose Random Forest as it performed the best. The detection and performance calculated based on img2 are shown below.

   ## Performance on image 2: ##
   
<img width="697" height="393" alt="Screenshot 2026-10-18 at 2 32 16 AM" src="https://github.com/user-attachments/assets/abaffae5-b829-450d-bddf-380e21dca342" />

  ## Evaluation of image 2: ##
   
<img width="449" height="133" alt="Screenshot 2026-10-18 at 1 47 04 AM" src="https://github.com/user-attachments/assets/606ddaf5-fd19-436b-bdae-85ee56d9804e" />

From the performance on image 2, we can also visually see that the ML algorithm performed worse at identifying barnacles than the actual labeled dataset. This is probably because we trained the model on only one data point, image 1. But if we add more data, accuracy will also increase. 
In the evaluation of image 2, we can see that the model’s accuracy is at 60%. Looking at the precision and recall, we can see that the recall is high, meaning it is able to detect most of the barnacles. However, the precision is low, meaning it also detects things that are not barnacles (e.g., background) as barnacles. To improve the system, we would have to improve the recall. Some possible ways could be to add more features for the barnacle characteristics or better tune the parameters. 

We also have other unseen images that we can use the model to predict the number of barnacles. However, we can't evaluate performance because we don’t have the labels (or masks) for them. Based on training on two data points (img1 and img2), the image below shows that my model predicted 821 barnacles for unseen image 1 and 96 barnacles for unseen image 2. Visually, we can see that it has performed well in these two images after being trained on two image datasets.
	
  ## Prediction on unseen images: ##
<img width="817" height="421" alt="Screenshot 2026-10-18 at 2 31 12 AM" src="https://github.com/user-attachments/assets/95e6e267-2f92-4f8e-a3e2-3b32ef2ab6c1" />


# 5. Conclusion

My prototype demonstrates that this semi-autonomous approach of combining computer vision techniques with machine learning is worth pursuing to automate parts of barnacle image analysis. We can see that by extracting features such as area, circularity, color, grayscale intensity, and edge density, the algorithm has key characteristics to distinguish between regions. However, it’s important to note that the performance varies across images and with the way the images are cropped using the selectROI method.
Instead of the scientists manually analyzing every region, the system could identify likely barnacle regions and prioritize uncertain predictions for human review.

Barnacles have distinct visual features, including circularity, color, brightness, and edge density, that can serve as useful data points for their identification. My current prototype has limited accuracy, but training on a larger dataset with more labeled examples could improve it. In order to scale this project, the accuracy would need to be increased. This can be done in various ways, such as human intervention, better training, and tuning.
Additionally, the need to manually crop the desired region in each image limits the system’s efficiency and scalability. Therefore, while the approach shows potential for automating barnacle identification, improving classification accuracy and automating region selection would be essential before it could reliably reduce scientists’ workloads.

To make this approach more scalable as the number of images (data points) increases, the system should automatically detect the desired region in each image rather than requiring manual cropping. Increasing the amount of labeled training data could also improve classification accuracy. Additionally, tuning the machine learning algorithms' hyperparameters and trying more advanced, contemporary machine learning models like BERT or ELM could further improve performance. Additionally, we can consider more factors that would be unique to a barnacle to improve the accuracy further, such as texture or aspect ratio of a barnacle.

Human feedback would also be crucial for refining the model over time. Once the model identifies potential barnacle candidates, a human can review the predictions, reject false positives, and add missed barnacles to improve future predictions. This would be especially important for barnacles clustered closely together and hard to detect. Human review could also help identify missed barnacles in blurry images or images with bad lighting. 


## Limitations

The approach may perform differently on different images due to variations in barnacle appearance, image quality, lighting, and background texture. Since the classifier can only identify candidates detected through OpenCV contour detection, it may miss barnacles with faint or overlapping edges. Also, it might overcount the number of barnacles due to irregular edges and nested boundaries. During training, I considered all candidates with at least 5% overlap with the provided mask to be potential barnacles. However, this threshold may cause background regions to be wrongly identified as barnacles if they overlap with the thick mask outlines. Rocks and other objects with similar visual characteristics may also be incorrectly identified. 

Additionally, the model was trained on only two annotated images, making it hard to learn all the different barnacle characteristics and reducing its reliability on images with different lighting conditions, textures, or barnacle distributions. Finally, the current implementation requires manual cropping of the region of interest for each image, which produces slightly different results each time and limits its scalability.


# 6. Learning Process
I greatly enjoyed working on this project and learned a lot of new things in the process. I was particularly excited to learn more about OpenCV functionality and see how it works. For example, I had not used the selectROI function before, but using it for this project made me realize its usefulness and importance, and I will implement it in future projects. 

Additionally, I greatly enjoyed learning about implementing machine learning with scikit-learn in Python. Previously, my experience with ML algorithms was mostly restricted to the R programming language. But through this project, I realized how accessible and fun Python libraries are, and how they can solve even more complex problems because of the wide array of features available in libraries like scikit-learn.

Lastly, I enjoyed the problem-solving and debugging aspect of this project. I started with a strong understanding of the problem and a clear idea of how I wanted the image analysis and machine learning parts to work together, which helped me lay out each step and task I wanted to implement. Additionally, understanding the meaning of performance metrics such as accuracy, precision, recall, and F1 score, along with understanding what high and low values indicate, enabled me to fine-tune parameters effectively.

I found this project to be very rewarding in terms of learning new tools and applying them to a real-world problem.
