# DALI Barnacles Challenge
This project uses computer vision and machine learning approach to automate barnacle image analysis. It uses image processing techniques to identify barnacle regions and extract features, then applies a machine learning classifier to distinguish the barnacle from the surroundings. 

# Files 
- Images: All the images given of the barnacles and the masks
- README: Project overview, setup instructions, and project documentation.
- trial.py — Main script that runs the image processing, feature extraction, classification, and visualization.
  

# Requirements 
- Python
- OpenCV (opencv-python)
- NumPy
- scikit-learn
- Matplotlib

# How to run 
1. Clone or download the repository.
2. Install the dependencies listed above.
3. Ensure the input images and any required reference masks are available at the paths specified in trial.py.
4. Run on terminal using "python trial.py"
5. When an image window appears, select the region of interest by dragging a blue square, then press Enter to continue. Repeat this process for additional images if given.

The trial.py file contains all the code to run this prototype. The output of this file provides the barnacle detection and accuracy for img2 as well as the predicted barnacle count and possible candidate barnacle highlighted for the two unseen images given (unseen_img1 and unseen_img2). 

# 1. Initial Brainstorming

The main problem presented by the scientist is the high effort and time it takes to manually count the barnacles in an image. Thus, we can automate this process using image analysis tools, such as OpenCV and machine learning algorithms. 

- Use OpenCV to break down the image by first cropping to the desired portion (inside the green border), 
- Determining the contours presented in the image 
- Develop a characteristic dataset with features like the shape, color, and edges
- Use the masked image to generate labels - determine if a contour is actually a barnacle or not depending on whether the masks has a barnacle there or not
- Once have the x (features) and y (labels) in the dataset has been created, I will implement Random Forest to train the model to determine whether something is a barnacle or not using the two datapoint (image 1 and image 2) 
- Test the ML model on the unseen images to determine the number of barnacles present

# 2. Subtasks

<img width="626" height="683" alt="Screenshot 2026-10-18 at 4 19 22 PM" src="https://github.com/user-attachments/assets/6d423f54-65fe-46e7-956c-5a20877edc10"/>

# 3. Specific subtask and its performance 

A key step in building this model is the machine learning classification model that will be used to classify future unseen images and determine the number of barnacles. The ML model will use the dataset of features and labels created by img1 and img 2 to train on whether something is a barnacle or not and use it to count barnacles in unseen images in later stages.

To evaluate the model's performance, I can use the classification_report function in the Python scikit-learn library. This gives important metrics such as accuracy, precision, recall, and F1 score. 

The accuracy would help me evaluate how well the model was trained and is performing on unseen data to gauge its usefulness. Precision will help determine the trustworthiness of our "yes" labels (whether the contours predicted to be barnacles are actually barnacles or whether the model incorrectly judges other objects as barnacles too).
Recall will help in determining how well the model finds all the correct "yes" cases (barnacles) present in the data. F1 score is a good metric because it balances recall and precision, since it is the harmonic mean of the two values. These would help guide parameter tuning depending on which value is low/high.

After running the model on unseen data, I can manually compare how many barnacles are highlighted with the actual number present in the image and use metrics (precision, recall, and F1 score) to assess overall performance.

Since we were only provided with two masked images, the machine learning algorithm has very limited data, so it will not perform that well. We can still calculate precision, recall, and F1 by training on img 1 and testing on img 2 because we can create output labels for both. However, this is very limited data, and having more would improve accuracy and lead to better predictions.

# 4. Prototype Analysis

After tuning a few parameters and testing a few ML algorithms, I chose Random Forest as it performed the best. The detection and performance calculated based on img2
 is shown below.

   ## Performance on image 2: ##
   
<img width="697" height="393" alt="Screenshot 2026-10-18 at 2 32 16 AM" src="https://github.com/user-attachments/assets/abaffae5-b829-450d-bddf-380e21dca342" />

  ## Evaluation on image 2: ##
   
<img width="449" height="133" alt="Screenshot 2026-10-18 at 1 47 04 AM" src="https://github.com/user-attachments/assets/606ddaf5-fd19-436b-bdae-85ee56d9804e" />

Focusing on the row with Barnacle detection performance, we can see that the model’s accuracy is not that high at 60%. Looking at the precision and recall, we can see that the recall is high, meaning it is able to detect most of the barnacles. However, the precision is low, meaning it also detects things that are not barnacles (e.g., background) as barnacles. 

We can also visually see that the ML algorithm performed worse at identifying abrnacles than the actual labeled dataset. This is probably because we trained the model on only one data point, image 1. But if we add more data, accuracy will also increase. 

We also have other unseen images that we can use the model to predict the number of barnacles in. However, we can't evaluate performance because we don’t have the labels (or masks) for them. Based on training on two data points (img1 and img2), my model predicted 821 barnacles for unseen image 1 and 96 barnacles for unseen image 2.
	
  ## Prediction on unseen images: ##
<img width="817" height="421" alt="Screenshot 2026-10-18 at 2 31 12 AM" src="https://github.com/user-attachments/assets/95e6e267-2f92-4f8e-a3e2-3b32ef2ab6c1" />


# 5. Conclusion
- Barnacles have specific features that can be used as data points for their identification, including their circularity, color, brightness, and number of edges. 
- To scale up this project, automation of a highly accurate model would be needed.
- A few issues with my model making it hard to pursue
	- Accuracy is too low (probably can be increased if more training data and labels are present)
	- Requires cropping to the desired region manually for each image

- Possible solution: manually cropping to the desired region for each image isn't realistic if the number of data points increases to thousands or millions.
	- We would need automatic detection of the cropped region for all training data by using the already cropped mask provided. Then cropping would only be required for additional unseen (test) images.
	- The greater the amount of data, the higher the accuracy!
	- Test tuning parameters on the machine learning algorithms to further improve accuracy, or try contemporary algorithms (e.g., BERT, ELM).

- My current approach works for a small amount of data where a human can look at all the images and crop to the area of interest. Additionally, we can consider more factors that would be unique to a barnacle to improve the accuracy further, such as texture or standard deviation of the colors and brightness. 
- Human feedback on the model would be crucial because it would help refine the model further as it learns from it. For example
	- Once the model predicts possible barnacle candidates, a human can review them and reject false positives or add missed ones. This is especially important for barnacles in close proximity that seem hierarchical and harder to detect.
	- For images with bad lighting or blurry images, a human can better identify missed barnacles.
	- If specific information about barnacle measurements or their clustering behavior is known, it could be added as an additional factor for the model to consider and test.

## Limitations

- The approach may perform differently across images due to variations in barnacle appearance, image quality, lighting, and background texture.
- The approach may miss barnacles with faint or overlapping edges because the classifier can only identify candidates detected by contour detection in OpenCV.
- It might be overcounting some of the barnacle because of irregular edges and nested boundaries.
- While training the ML model, I considered the overlap of 5% for each candidate with the mask provided but this might accidentally think of the background as a barnacle if it overlaps with the thick outlines. Also other rocks similar to barnacle might be counted as barnacle.
- The current code requires manual interaction during image cropping, so it is not yet fully automated.
- Limited training data: it is trained on only two annotated images. So, it might not be able to fully understand all the barnacle features well, making it less reliable on images with different lighting, textures, or barnacle distributions.

# 6. Learning Process
I greatly enjoyed working on this project and learned a lot of new things in the process. I was particularly excited to learn more about OpenCV functionality and see how it works. For example, I had not used the selectROI function before, but using it for this project made me realize its usefulness and importance, and I will implement it in future projects. 

Additionally, I greatly enjoyed learning about implementing machine learning with scikit-learn in Python. Previously, my experience with ML algorithms was mostly restricted to the R programming language. But through this project, I realized how accessible and fun Python libraries are, and how they can solve even more complex problems because of the wide array of features available in libraries like scikit-learn.

Lastly, I enjoyed the problem-solving and debugging aspect of this project. I started with a strong understanding of the problem and a clear idea of how I wanted the image analysis and machine learning parts to work together, which helped me lay out each step and task I wanted to implement. Additionally, understanding the meaning of performance metrics such as accuracy, precision, recall, and F1 score, along with understanding what high and low values indicate, enabled me to fine-tune parameters effectively.

I found this project to be very rewarding in terms of learning new tools and applying them to a real-world problem.
