# darrenyanggg.github.io

## Proposal 
The dataset I’m using is Yelp Complete Open Dataset. I used Kaggle to find this dataset and the link is https://www.kaggle.com/datasets/adamamer2001/yelp-complete-open-dataset-2024. In this dataset, 5 JSON files make up the entirety of the dataset.

The business.json file contains business data which tells us the business ID, name, full address, longitude and latitude, the star rating, number of reviews, whether the business is open or closed, attributes, and categories of the business. The user.json has the user ID, name of the user, number of total reviews they have written, the date they joined Yelp, their friends on Yelp, the amount of useful, funny, and cool votes sent by the user, the number of fans they have, the years they were elite, average star rating of all their reviews, and all the metadata associated with the user’s received compliments. The checkin.json file has the business ID and the timestamps of checkins on a business. The tip.json file has the business ID and user ID, the tip written by a user to the business (shorter than reviews), the date the tip was written, and the number of compliments that tip received. The review.json includes the review ID, the user ID that wrote the review, and the business ID that the review is aimed at. It also includes the star rating, the date on which the review was written, what the review said, and the number of votes the review got for its usefulness, funniness, and coolness. 

I would like to predict a business's star rating based on the various information provided by its users on Yelp. I would predict the exact column inside the review.json and its stars variable. I would probably use random forest to predict this, as the main focus would be using the users’ reviews and their features to predict a review star for each user. 



## Data Acquisition
I downloaded the yelp-complete-open-dataset-2024.zip data file by enabling the Kaggle API token and using its API command: kaggle datasets download -d adamamer2001/yelp-complete-open-dataset-2024. I then unzipped the file with the command: unzip yelp-complete-open-dataset-2024.zip. This gave me two folders: yelp_dataset and yelp_photos, confirming that the dataset files are unzipped as all the files I need are inside yelp_dataset (with yelp_photos as another dataset made up of jpegs from Kaggle). 

I then used the ‘gcloud’ command to make a new bucket called yelpfrog inside my darrentutorial project in Google Cloud Storage and copied my data files to the landing folder. Since the dataset has more than one file, I used the --recursive option for the gcloud command to copy all files. The following command is what made all this possible: gcloud storage cp --recursive yelp_dataset/gs://yelpfrog/landing/.



## Exploratory Data Analysis and Data Cleaning
### Exploratory Data Analysis
From my Exploratory Data Analysis, I looked through all 5 datasets within the Yelp Complete Open Dataset. 

Firstly I found that of the 5 only the business.json had missing values while the other 4 didn’t have any. The business dataset had a total of 150346 observations and had missing values in its attributes, categories, and hours columns. Through summary statistics, I found that in the review_count column, the average number of reviews is approximately 44.87 with a standard deviation of approximately 121.12. The lowest number of reviews is 5 while the highest is 7568. In the stats column, the average star rating was about 3.6 with a standard deviation of 0.97. As expected the lowest star rating is 1.0 and the highest is 5.0. I also found that the address 185E State St appeared the most (minimum isn’t shown as most addresses appear once). For categories, the least popular category is 3D Printing, Local Services, Hobby Shops, and Shopping while the most popular ones are Zoos, Tours, Arts & Entertainment, Hotels & Travel, and Active Life. The most frequented city is Lithia and the least frequent is AB Edmonton. The least popular business is Grow Academy whereas the most popular is called Transformational Abdominal Massage by Jada Delaney. The most popular state is XMS (I have no clue where that is) and the least popular is in AB. Through the use of visualizations, I found that there are much more opened businesses than closed ones. I also found that the distribution of stars, weighted by the number of reviews,  is left-skewed with the peak at around 3.7. There seems to be a gap between 2.6 and 2.9 which may be numbers people typically don’t choose when rating a business. 

For the checkin dataset, there are a total of 131930 observations. The first check-in was on 2009-12-30 at 02:53:… while the most recent check-in was on 2022-01-19 at 01:15:21. 

In the review dataset, there are 6990280 observations. Through summary statistics, the average star review is 3.75 with a standard deviation of 1.48. Like the business star, the min and max are at 1 and 5. What sets this apart from the business star column is that this one is more accurate due to it being individualized based on review rather than totals in a business. I also found that the first review was on 2005-02-16 at 03:23:22 and the most recent being 2022-01-19 at 19:48:45. Also, the minimum review length is 1 word while the most is 3079 with an average length of review being 105.80. 

The tip dataset has 908915 observations. The first tip written by a user was on 2009-04-16 at 13:11:49, with the latest one being 2022-01-19 at 20:38:55. The minimum compliment_count is 0, with 6 being the highest. Due to this, the average compliment_count is only 0.01, with a standard deviation of 0.12. As for the word count of a tip, the minimum number of words was 1, while the most was 225, with the average being 11.3.

Finally, in the user dataset, there are a lot of numeric variables. Only focusing on what’s important, the number of reviews an average user writes is 23.39 with a standard deviation of 82.57 due to the maximum being at 17473 (minimum is 0). A user’s average rating is 3.63 with a standard deviation of 1.18. I also found that the earliest user started on 2004-10-12 at 08:46:11 with the most recent being 2022-01-19 at 17:15:47

### Data Cleaning 
For the business dataset, I cleaned it by first filling in the missing values in the categories column with the string “General Business” as the column is important when it comes to categorizing the type of businesses. I then dropped the attributes m, hours, latitude, and longitude columns as they weren’t needed. I also changed the is_open column to a Boolean data type because only 2 possible values are making Boolean ideal. 

For the checkin dataset, I changed the date’s datatype to timestamp; since it was small, that was all I needed to do.

I changed the date’s data type to the timestamp for the review dataset and added a word_count column based on the text column. I also dropped the cool and funny columns as they weren’t relevant. 

In the tip dataset, I changed the compliment count’s datatype to integer and date’s to date. I also added a word_count column based on the text column. 

Finally, in the user column, I dropped the compliment_cool, compliment_cute, compliment_funny, compliment_hot, compliment_list, compliment_more, compliment_note, compliment_photos, compliment_plain, compliment_profile, compliment_writer, cool, elite, friends, and funny column because they weren’t needed. I changed the yelping_since datatype to date. Also, I decided to change the datatype of average_stars, fans, review_count, and useful columns to integers. 

Some challenges I would have in feature engineering are writing the code and using all these new libraries/functions that I know in theory but not in usability. For example, I know TensorFlow is for machine learning, but when I tried it, I couldn’t even load it. I also heard the modules keep updating making some stuff obsolete. 



## Feature Engineering and Modeling
I first joined a business and reviewed data frames together as I wanted to utilize review columns such as text and its stars (the label) while also using information about the business like the name, city, state, and review_count. 

After this, I created a bucketed review_count into 5 buckets as I remembered this column had outliers since the average review_count was 44.87 and had a standard deviation of 121.12 while its minimum and maximum were 5 and 7568 respectively. I then string-indexed city and state so I could one-hot encode them along with the bucketed review_count so they could go to the next step of vector assembling. Afterwards, I performed sentiment analysis on text and to make sure pipelines would integrate sentiment_analysis_udf I had to transform it from a function giving us sentiment_transformer. Finally, I vector assembled all the encoded columns and sentiment_score to return a combined features column.

I then created my model estimator utilizing RandomForestRegressor specifying the label, features, and what I want my prediction to be. With that, I created my pipeline to read this estimator and passed it through business_review_features (spark data frame with all the original data and features made along the way). I then created a separate data frame containing only the key columns from the original data frame, consolidated vectorized features, and the label (review_stars). With that, I created my train and test data using the random split method for a 70/30 split.

To perform cross-validation, I created a regression evaluator to get RSE, R2, and RME, etc. I then built the parameter grid to create a cross-validator using the hyperparameter grid specifying that I wanted 3 folds. Afterward, I trained the models and showed the average performance over the 3 folds. As a result, I got 1.0951429806404178 as my training RMSE. 

Now that I trained my model I got the best model from all the models I’ve trained to use to predict the test set. With that, I calculated the RMSE again along with R2 and MAE. The results I got back told me that R2 = 0.45509332808793435, RMSE = 1.0914973653215911, and MAE = 0.8863864133283145. My R2 explains how well my model fits the data and explains that 45.5% of the variability in my review_stars variable is explained by my features. RMSE is 1.09 which tells us that on average, my model predictions deviate from the actual values by about 1.09 units. As for MAE, it is 0.88 which, like RMSE, tells us that my model predictions deviate from the actual values by about 0.88 units. The calculated R2 and RMSE are moderate values while the MAE is low indicating that this model is a bit below average when it comes to accurate predictions. 

One challenge I had was ensuring my features were made right and weren’t too big when they were all vectorized. I also encountered a problem with my sentiment analysis twice: the first time I couldn’t get the text module to be installed in all clusters. I solved this by simply making one big master node instead of 1 master and 2 worker nodes. The second issue was the fact that pipelines don’t run functions so I had to convert my sentiment analysis udf function into a readable one by utilizing Transformer which allows for sentiment score to be integrated into the pipeline. When I first started I made my pipeline contain all the features I created along with the regression variable that contained the regression function. I then found out that that was what made my model never run because I was trying to fit every feature I created and pass it through the regression model. This made me opt to transform the functions early on and make the pipeline only contain the regression variable.

As I tried to make a new model, I found out sentiment_score, the feature I was most confident in had no importance to the model which probably made sense since reviews are individual feelings on the whim. At the same time, sentiment is a generalized feeling made over time. While creating the new model I ran into the problem of trying to encode name and address variables even though I properly indexed them, meaning something within the variable is abnormal (not missing any values as I’ve checked that. So I concluded that this model, while bad is the best I can come up with. Since I was trying to predict the review rating of individuals who rated them through Yelp, I realized that unless I have many features relating to customer behavior and mood, I won't be able to make a better model as I’m essentially trying to predict how each person who rates a business based on how they feel at a particular time and place. 

I would change the label to business star reviews instead of the initial individual review stars. In that case, the model performs better as overall business ratings are much easier to predict than individual ratings made by individuals. Although R2 = 0.2190848290537395 meaning only 21% is explained by the model, we get an RMSE = 0.8292076311018942 and MAE = 0.6769763638259244 which are both lower than the original model, meaning they deviate less from the actual values. Unlike the previous model, this model didn’t experience a Division By Zero error due to the true positive and false positive being 0. This meant that only 1 metric out of the 4, accuracy had a value that meant something. In this model I found that accuracy was 55%, precision was approximately 44.44%, the recall was 50%, and the F1 score was 47.06%. All of this told me that this model wasn’t good but not bad, as more than half of the results were accurate. 

After some feedback, I realized I needed random forest regressor hyperparameters in Model 2 (business review model). This was why for this model I added 10 trees with a maximum depth of 3. I chose to make the hyperparameters simplistic to save time and not run into any PySpark disk-related errors even though the ideal hyperparameters should’ve had multiple trees and depths to form combinations of hyperparameters to allow for a more thorough search of the parameter space. The results this time were different while some were similar. R2 = 0.20234340852099342 making less of the model being explained than the other 2 models, we get an RMSE = 0.8380488594865366 and MAE = 0.6817354983882484 which is better than Model 2 by approximately 1% each meaning the deviation is even less from the actual values. Now the biggest difference is the accuracy, precision, recall, and F1 scores as they have skyrocketed to numbers near 1 or 100%. Accuracy has become 81.60% and precision is a bit higher at 81.96%. Now recall is the most interesting as it became 99.45% which is extremely close to 100% and the F1 score is 89.86%. These results tell us that the true and false values of our new model showed better results and with a bit more tuning our metrics may become even better too. 



## Visualization
### Model 1 (Individual Ratings):
I made 4 graphs visualizing my model. The first one we see is a scatter plot of predicted values compared with the actual ones. The dotted line indicates the success of actual and prediction and as we see this isn’t a very good model. It doesn’t follow the linear line we created telling us that our prediction didn’t follow much of any rules. The only one I can spot is that our predictions tend to lie within the 2.0 to 4.5 range which is probably where the majority of our actual values fall in. 

On the next graph, I visualized the distribution of error. A great graph would look like a bell curve and as we see in our graph it is a bit wonky and a bit left skewed indicating a bias towards lower rating. 

The box plot is interesting since the median is about the same when we look at prediction and actual values. When we look at the other values however we see that their minimum and maximum values don’t go as far as what the actual values are meaning our predictions are centralized towards the median. 

The residual plot shows that the model is not a good fit for the data since there is a very clear pattern in the graph. This tells us that the prediction itself is missing key information or features that aren’t there. There are also a few gaps in the 3.0 to 3.5 range indicating that the model can’t predict this clearly due to insufficient features or data. 

### Model 2 (Consolidated Business Ratings):
Here, we see the same thing as the previous model, but it's better because it’s not as spread out. The range of the predictions only goes from 3.0 to 4.0, meaning there’s a feature that may be making it biased towards the upper range of ratings. 

If we look at the distribution of errors visually we can see that it looks more normalized than the previous one and there’s a clear shape. Although it does drop significantly at the upper bound I can safely assume that most of the errors are clustered around the mean and there are few outliers. 

Now, the box plot for the prediction is more centralized at the upper range, with outliers at the lower end of the box plot at 3.0. This tells me that the business review models have a clear bias towards high ratings and do not take into account the low ratings enough. 

The residual plot shows that the model is not a good fit for the data since there is a very clear pattern in the graph. This tells us that the prediction itself is missing key information and features. 

### Model 3 graphs are very similar to Model 2 with only some minute differences



## Summary And Conclusion
After creating my regression pipeline using random forest I realized that the label I was trying to predict was too broad. Instead of predicting individual ratings made by each person, the label I should’ve been focused on was the business itself and its corresponding rating made by everyone who rated that business. This gave me better results than the individual ratings as shown in my model and visualizations. Also since I didn’t put any hyperparameters that distinguished a Random Forest Regressor from any other regression, I effectively made my regression model linear instead of random Forest as I hoped for. I went ahead and gave it those hyperparameters in my third model and got the worst metrics but extremely better results when it came to confusion matrix-related computations. All in all, it seems that a lot of features are required to predict business ratings and when doing so it must be done using a nonlinear regressor like Logistic regression or Random Forest. When doing so we need to play around with the hyperparameters to find the best ones and make sure we aren’t moving down a linear route. 
