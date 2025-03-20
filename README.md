# deeplearning23
Deep Learning Project - Skin Cancer Detetion
INTRODUCTION
 The Portuguese Skin Cancer Association estimates that skin cancer claims the life of one Portuguese
 person every day on average. It is also known that early detection of skin cancer increases the likelihood of
 recovery by more than 90%.
 By working on this project, we hope to demonstrate our skills in deep learning by creating a neural network
 model that can recognize various forms of skin cancer based on images.
 Our aim is to illustrate based on what we learned in the classes how advancements in technology can have
 a good effect on multiple domains, including health, and prolong the life expectancy of individuals that
 developed a type of skin cancer based on early detection.
 PYTHON LIBRARIES
 During our project, TensorFlow and Keras were crucial to the development of the neural networks, and Keras
 Tuner was utilized for hyperparameter tuning. For Visualization tasks, we used matplotlib, seaborn and
 mpl_toolkits.mplot3d which aided us to see insights into the dataset characteristics. Scikit-Learn also
 contributed to the preprocessing of data and the evaluation of the model's performance through tools like
 OneHotEncoder and f1_score. We also used pandas and numpy for data manipulation.
 Furthermore, the os library helped to made file and directory operations easier and opencv was also utilized.
 METADATA PRE-PROCESSING
 We start by importing our dataset and creating a new column named “path” with the respective path for each
 image in our train folder. 
We then checked for missing values and we concluded that “age” was the only column with missing values
 (57). In the age variable, we also checked that 39 rows had age equal or less than zero. Facing this we
 decided to fill all the missing values and zero values with the median of the ages of our dataset. We checked,
 using the “image_id” column, if all the rows in our dataset were linked up with an image in our train folder, if
 not we would drop those rows. That left us with 7511 rows out of 10015.
 We noticed that  the distribution of our target column “dx” was represented by seven types of skin cancer: -“bkl”(benign keratosis)-“nv”(melanocytic nevi)-“df”(dermatofibroma)-“mel”(melanoma)-“vasc”(vascular)-“bcc”(basal cell carcinoma)-“akiec”(actinic keratosis)
 The “nv” type of skin cancer represents 5029 out of 7511 rows in our dataset. This led us to conclude that
 our target variable was an imbalanced variable.
 Analysing our other charts we could conclude that the “localization”, “age” and “dx_type” variables were also
 imbalanced. The variable “localization” represents for each image the localization where the potential danger
 sign of skin cancer is located. The “back” and the “lower extremity” were the areas where the images were
 located more frequently, representing around 45% of our dataset. The least frequently localizations were
 “chest”, “foot”,”neck”, “scalp”,” ear”, “genital”, “acral” and the images where the localization was “unknown”,
 representing around 10% of our dataset. The other 45% were distributed between “trunk”, “upper
 extremity”,”abdomen” and “face”. In the ” dx_type” variable 4014 out of 7511 of our rows were represented in
 the “histo” class, following the “follow_up” with 2775, “consensus” representing around 673 of the patients
 and then “confocal” with 49.
 Page 1
In the “age” variable most of the patients were included between 30 
and 60 years of age. Being the other patients located in the range of
 5 to 30 and from 60 to 85 years old. 
The variable “sex” was reasonably balanced but with around 500
 more men than women. We also had 44 rows with “unknown” sex. 
By all that we can state that our dataset is imbalanced.
 We also performed one hot encoding in our other variables: “
 dx_type”, ”sex”, and “localization”. Instead of dropping or imputing the
 unknown values in the “localization” and “sex” columns we decided to include them in the one hot encoding, by
 doing this we could guarantee that no important information was being discarded neither we were predicting
 outliers.
 To finalize the pre-processing of our CSV we decided to normalize our variable “age”, using MinMaxScaler.
 We did that to converge the scale of our data and to ensure quickness in our model. 
IMAGE PRE-PROCESSING
 The first thing we did was check the size of our biggest and smallest images. We concluded that all
 our images had the same size (600,450).
 In our image pre-processing was our objective to remove the hairs of our images. We found out
 that many of the images had the presence of hair and we formulated a thought that these hairs
 could affect the performance of our model. 
First, we created a new column in our dataset named “image_remove_hair” and applied a
 remove_hair function that included the conversion of our images to arrays, grayscale and binary
 masks. We then resized our images to the size (50,50) and compared our figures before and after
 the hair being removed and we concluded that our objective was successfully achieved.
 INITIAL MODEL WITHOUT SKIN HAIR REMOVE FUNCTION
 We started our approach in the models by creating a model without the removal of the skin hair function
 applied. Our goal was to create two models, one with the function applied and one without it, to conclude if
 this process of removing the skin hair will provide better results.
 We choose not to incorporate all the CSV file columns in our neural network model, opting for an approach
 that relies on image data. This decision supports a focus on the deep learning model capacity to directly
 learn meaningful features from images, enhancing simplicity and interpretability while maintaining
 computational efficiency. Furthermore, we also decided to encode our target column “dx” and introduce it in
 the model instead of using one-hot encoding, prioritizing model simplicity, reduced dimensionality, and
 improved adaptability in recognizing the connections among various classes. We resized our images in the
 pre-processing to (50,50) so our input shape in the model was (50,50,3), where the number 3 represented
 the RGB colour system. We also normalized (/256) them and we performed a train test split with stratify
 because our target column was imbalanced as we already have seen before. We also performed data
 augmentation using ImageDataGenerator to increase the size of our training set. Getting the train and data
 generator we defined the batch size as 32, following this, after every 32 interactions our model should
 update the weights. 
Page 2
Our approach started with the construction of a CNN model as we have seen in practical classes, with three
 convolutional layers with the rectified linear unit activation function and filters size of 32, 64 and 128
 respectively. Between each of our convolutional layer we applied 2D Max pooling, essentially, to reduce the
 spatial dimensions and to capture the most important features. We then used flatten and 2 fully connected
 layers with 512 and 7 neurons respectively. The 7 neurons are the number of classes of our target and
 being the last layer we applied the “softmax” activation function. Between the 2 fully connected layers we
 applied a dropout to avoid overfitting. We compile our model using “Adam” optimizer, as metric “accuracy”
 and as loss function “categorical crossentropy”, because we are facing a multi-class classification problem.
 To reach fair conclusions about the model we then performed the f1-score, more precisely we focused on
 the weighted f1 score because of the fact that is better to evaluate the model ability to perform well across all
 classes, regardless of their size. F1 score is a much more important metric to our project than accuracy, due
 to the fact that as we concluded before our dataset is imbalanced. The result of the weighted f1-score on the
 validation set of this initial model without removing skin hair was of 0.7417.
 .
 INITIAL MODEL WITH SKIN HAIR REMOVE FUNCTION
 After we have performed the model without the remove skin hair function, we proceeded to use the same
 model but this time with the images with the function applied.
 We followed the exact same approach, we converted the images to numpy arrays, normalized them and
 performed the train test split with the stratify to our target.
 We used the same data generator and the exact same model that we used in the previous step.
 After assessing the model performance, the weighted F1-score on the validation set was 0.7473  and
 consequently was concluded that applying the skin hair removal preprocessing step could slightly increase
 the model performance compared to the model performance without employing this step. So, we decided to
 maintain the transformation of removing skin hair for our final model. 
KERAS TUNER RANDOM SEARCH AND FINAL MODEL
 As we analysed before, our initial model had a decent weighted f1-score, but we wanted to improve starting
 from there. So, we performed a Random Search with Keras Tuner to find the final structure of our CNN model.
 We tried multiple combinations (25 for each) with 3 and 4 layers because we wanted to maintain a similar
 structure from our initial model. For the search with 3 layers with used a dropout in the dense layer from 0.2 to
 0.6 and the layers had between 16 and 256 units. The search with 4 layers was similar, and in both of them
 we used two dense layers to capture the output, one starting with 128 to 512 units and another with 7. We
 then checked the accuracy in the validation for every model and displayed the best models in order of
 accuracy, due to the fact that we could not do it by the f1-score in the Keras Tunner. From that output we
 started to train multiple models of the Random Search until we found the model with the best performance.
 From the 4 layer model, this were our best results:
 Page 3
From the 3  layer model , this were our best results:
 From the best accuracies found from the searches, we tried multiple
 models, and based on the performance of the f1 we decided that the
 search wasn´t helpful  because we got similar or worse results, so
 we decided to use our initial model, displayed here:
 While the Keras Tuner did not yield a better-performing model, we
 decided to include the details of our exploration and the attempted
 variations of the model in the report and notebook. This decision was
 made to provide an overview of the model selection process and to
 highlight the importance of acknowledging both successful and
 unsuccessful attempts in the pursuit of optimal model performance.
 PRE TRAINED MODEL - DENSENET121
 Transitioning from making our own model using Keras Turner, we incorporated DenseNet121, a pre-trained
 deep neural network architecture. The inclusion of DenseNet121 served two purposes, firstly it provided a
 benchmark for comparing our model’s performance. This comparative analysis helps us to assess the
 effectiveness of our custom model against a well established one, offering us valuable insights into the
 strengths and weaknesses of both models. After running the model with 50 epochs we decided to readjust it
 to 20 epochs to avoid overfitting. It also allows us to understand the potential advantages of using a
 sophisticated pre-existing model,  leading to improvements in metrics such as F1-Score. The  F1 Score of
 the validation set was 0.7540.
 . 
PREDICTING ON TEST DATA 
Finally, after all the other pre steps of our project, we tested our test data in our pre-trained model and final
 model. To perform that we applied the same preprocess that we applied in our training and validation data to
 the test. We then predict the test images, first with the DenseNet121 pre-trained model and then with our final
 model. We concluded that the results on the f1 score were 0.7455  with our custom model and 0.78 with the
 DenseNet121. A confusion matrix was also done with the results of our custom model so that we could better
 identify our most common errors and strengths. Upon examination, we found that the cancer type “nv” was
 notably our most accurate predicted type of cancer, showing us a high level of proficiency.
 Page 4
This outcome can easily be attributed to the large amount of training images that were available for training
 for this specific type of cancer (5029 out of 7511), providing the model with suitable information for learning.
 In contrast, other types of cancer showed lower correct predictions, with a special focus on the “df” cancer
 type where we never even got a single right prediction. This discrepancy occurs from the lack of training
 data that we have for “df” since we only had 86 images. Taking note of this constraint, in our future projects
 we should try to have more diverse training data or explore ways to augment the significance of the classes
 with fewer images, in the hope that this approach improves the model’s performance over all the different
 classes, especially in scenarios with sparse training data.
 Accuracy and F1 score of our final model on test data 
Examples of incorrect predictions:
 Incorrect Predictions of “df” type of cancer:
 Correct Predictions of “bkl” type of cancer:
 CONCLUSION
 As we can observe from the evaluation of the test data, the pre trained model DenseNet had slightly better
 performance in all metrics  while our model achieved worse but still meaningful results. The fact that we
 could not find a better model with keras tuner after our initial try left us with the idea that for our
 preprocessing techniques and data, the model was well structured. Furthermore, we could have delved
 deeper into image preprocessing steps to potentially enhance performance further. It is worth noting that,
 although not explicitly discussed in the report and notebook, we experimented with various techniques such
 as image enhancement and contrast enhancement, but these approaches did not yield satisfactory results.
 The project was particularly rewarding due to its inherent challenges, and we appreciate the valuable
 insights gained through the exploration of different models and preprocessing methods. While we may not
 have achieved the desired improvements within the given timeframe, the experience has enriched our
 understanding of the intricacies involved in image classification tasks
