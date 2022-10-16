#!/usr/bin/env python
# coding: utf-8

# In[68]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


import pandas as pd


# In[6]:


get_ipython().system('pip install surprise')


# In[7]:


# Used to ignore the warning given as output of the code
import warnings                                 
warnings.filterwarnings('ignore')

# Basic libraries of python for numeric and dataframe computations
import numpy as np                              
import pandas as pd

# Basic library for data visualization
import matplotlib.pyplot as plt     

# Slightly advanced library for data visualization            
import seaborn as sns                           

# A dictionary output that does not raise a key error
from collections import defaultdict             

# A performance metrics in surprise
from surprise import accuracy

# Class is used to parse a file containing ratings, data should be in structure - user ; item ; rating
from surprise.reader import Reader

# Class for loading datasets
from surprise.dataset import Dataset

# For model tuning model hyper-parameters
from surprise.model_selection import GridSearchCV

# For splitting the rating data in train and test dataset
from surprise.model_selection import train_test_split

# For implementing similarity based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic

# For implementing matrix factorization based recommendation system
from surprise.prediction_algorithms.matrix_factorization import SVD

# For implementing cross validation
from surprise.model_selection import KFold


# In[8]:


rating = pd.read_csv('/content/drive/MyDrive/ratings.csv')


# In[ ]:





# In[9]:


rating.info()


# In[10]:


# Dropping timestamp column
rating = rating.drop(['timestamp'], axis=1)


# In[17]:


# Question 1: Exploring the dataset


# In[11]:


#Q 1.1 Print the top 5 rows of the dataset
rating.head(5)


# In[12]:


#Q 1.2 Describe the distribution of ratings. 
plt.figure(figsize = (12, 4))
sns.countplot(x="rating", data=rating)

plt.tick_params(labelsize = 10)
plt.title("Distribution of Ratings ", fontsize = 10)
plt.xlabel("Ratings", fontsize = 10)
plt.ylabel("Number of Ratings", fontsize = 10)
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.show()


# In[20]:


# As per Histogram, Rating '4' has highest count of ratings, about 28000K. Rating '3'is second with about 21000k
# and rating  '5' is third in count of ratings with about 15000k. The ratings are biased towards 4 and 3 more
# than the others.


# In[13]:


# Copying the data to another DataFrame
df=rating.copy()


# In[ ]:


# Q 1.3 What is the total number of unique users and unique movies?


# In[14]:


# Finding number of unique users
rating['userId'].nunique()


# In[15]:


# Finding number of unique movies
rating['movieId'].nunique()


# In[ ]:


# Q 1.4 Is there a movie in which the same user interacted with it more than once?


# In[16]:


rating.groupby(['userId', 'movieId']).count()


# In[17]:


rating.groupby(['userId', 'movieId']).count()['rating'].sum()


# In[ ]:


# The sum is equal to the total number of observations which implies that there is only interaction between
# a pair of items and a user.


# In[ ]:


# Q 1.5 Which is the most interacted movie in the dataset?


# In[18]:


rating['movieId'].value_counts()


# In[ ]:


# The movieID 356 has been interacted by most users which is 341 times.
# But still, there is a possibility of 671-341 = 330 unique users in our datasets. For those 330
# remaining users, we can build a recommendation system to predict who is most likely to interact with them.


# In[19]:


# Plotting distributions of ratings for 341 interactions with movieid 356 
plt.figure(figsize=(7,7))

rating[rating['movieId'] == 356]['rating'].value_counts().plot(kind='bar')

plt.xlabel('Rating')

plt.ylabel('Count')

plt.show()


# In[ ]:


# We can see that this item has been liked by the majority of users, as the count of ratings 4 and 5 is higher than the count of other ratings.
# There can be items with very high interactions but the count of ratings 1.0 and 1.5 may be much higher than 4 or 5 which would imply that the item is disliked by the majority of users.


# In[ ]:


# Q 1.6 Which user interacted the most with any item in the dataset?


# In[20]:


rating['userId'].value_counts()


# In[ ]:


# The user with userID:547 has interacted with most number of items i.e 2391.
# But still, there is a possibility of 9066-2391 = 6675 more interactions as we have 9066 unique movies in our dataset.
# For those 6675 remaining movies, we can build a recommendation system to predict which items are most likely to be watched by this user.


# In[ ]:


# Q 1.7 What is the distribution of the user-movie interactions in this dataset?


# In[21]:


# Finding user-movie interactions distribution
count_interactions = rating.groupby('userId').count()['movieId']
count_interactions


# In[22]:


# Plotting user-movie interactions distribution
plt.figure(figsize=(15,7))

sns.histplot(count_interactions)

plt.xlabel('Number of Interactions by Users')

plt.show()


# In[ ]:


# The distribution is skewed to the right. Few users interacted with more than 500 movies.


# In[ ]:


# QUESTION 2: CREATE RANK-BASED RECOMMENDATION SYSTEM


# In[23]:


# Calculating average ratings
average_rating = rating.groupby('movieId').mean()['rating']

# Calculating the count of ratings
count_rating = rating.groupby('movieId').count()['rating']

# Making a dataframe with the count and average of ratings
final_rating = pd.DataFrame({'avg_rating':average_rating, 'rating_count':count_rating})


# In[24]:


final_rating.head()


# In[ ]:


# Now, let's create a function to find the top n items for a recommendation based on the average ratings of items. We can also add a threshold for 
#a minimum number of interactions for a item to be considered for recommendation.


# In[25]:


def top_n_movies(data, n, min_interaction=100):
    
    #Finding movies with minimum number of interactions
    recommendations = data[data['rating_count'] >= min_interaction]
    
    #Sorting values w.r.t average rating 
    recommendations = recommendations.sort_values(by='avg_rating', ascending=False)
    
    return recommendations.index[:n]


# In[ ]:


# We can use this function with different n's and minimum interactions to get movies to recommend
# Recommending top 5 movies with 50 minimum interactions based on popularity


# In[26]:


list(top_n_movies(final_rating, 5, 100))


# In[ ]:


# Recommending top 5 movies with 200 minimum interactions based on popularity


# In[27]:


list(top_n_movies(final_rating, 5, 200))


# In[ ]:


# Now that we have seen how to apply the Rank-Based Recommendation System, let's apply the Collaborative Filtering Based Recommendation Systems.


# In[ ]:


# Model 2: User based Collaborative Filtering Recommendation System


# In[28]:


# Making the dataset into surprise dataset and splitting it into train and test set
# Instantiating Reader scale with expected rating scale
reader = Reader(rating_scale=(0, 5))

# Loading the rating dataset
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)

# Splitting the data into train and test dataset
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# In[29]:


# Build the first baseline similarity based recommendation system using cosine similarity and KNN
# Defining Nearest neighbour algorithm
sim_options = {'name': 'cosine',
               'user_based': True}
algo_knn_user = KNNBasic(sim_options=sim_options,verbose=False)

# Train the algorithm on the trainset or fitting the model on train dataset 
algo_knn_user.fit(trainset)

# Predict ratings for the testset
predictions = algo_knn_user.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)


# In[ ]:


# Q 3.1 What is the RMSE for baseline user based collaborative filtering recommendation system?
# As we can see from above, these baseline model has RMSE=0.99 on test set, we will try to improve this number later by using GridSearchCV tuning 
# different hyperparameters of this algorithm


# In[ ]:


Q 3.2 What is the Predicted rating for an user with userId=4 and for movieId=10 and movieId=3?


# In[30]:


# Let's us now predict rating for an user with userId=4 and for movieId=10
algo_knn_user.predict(4, 10, r_ui=4, verbose=True)


# In[ ]:


As we can see - the actual rating for this user-movie pair is 4 and predicted rating is 3.62 by this similarity based baseline model


# In[31]:


# Let's predict the rating for the same userId=4 but for a movie which this user has not interacted before i.e. movieId=3
algo_knn_user.predict(4, 3, verbose=True)


# In[ ]:


As we can see the estimated rating for this user-item pair is 3.2 based on this similarity based baseline model


# In[32]:


# Remove _______ and complete the code

# Setting up parameter grid to tune the hyperparameters
param_grid = {'k': [20, 30, 40], 'min_k': [3, 6, 9],
              'sim_options': {'name': ['msd', 'cosine'],
                              'user_based': [True]}
              }

# Performing 3-fold cross validation to tune the hyperparameters
grid_obj = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)

# Fitting the data
grid_obj.fit(data)

# Best RMSE score
print(grid_obj.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print(grid_obj.best_params['rmse'])


# In[ ]:


# Once the grid search is complete, we can get the optimal values for each of those hyperparameters as shown above.
# Below we are analysing evaluation metrics - RMSE and MAE at each and every split to analyze the impact of each value of hyperparameters


# In[33]:


results_df = pd.DataFrame.from_dict(grid_obj.cv_results)
results_df.head()


# In[34]:


# Now, let's build the final model by using tuned values of the hyperparameters, which we received by using grid search cross-validation.
# Using the optimal similarity measure for user-user based collaborative filtering
# Creating an instance of KNNBasic with optimal hyperparameter values
similarity_algo_optimized_user = KNNBasic(sim_options=sim_options, k=40, min_k=6,Verbose=False)

# Training the algorithm on the trainset
similarity_algo_optimized_user.fit(trainset)

# Predicting ratings for the testset
predictions = similarity_algo_optimized_user.test(testset)

# Computing RMSE on testset
accuracy.rmse(predictions)


# In[ ]:


We can see from above that after tuning hyperparameters, RMSE for testset has reduced to 0.98 from 1.05. We can say that we have been able 
to improve the model after hyperparameter tuning


# In[ ]:


Q 3.4 What is the Predicted rating for an user with userId =4 and for movieId= 10 and
 get_ipython().set_next_input(' movieId=3 using tuned user based collaborative filtering');get_ipython().run_line_magic('pinfo', 'filtering')


# In[35]:


# Let's us now predict rating for an user with userId=4 and for movieId=10 with the optimized model
# Remove _______ and complete the code
similarity_algo_optimized_user.predict(4,10, r_ui=4, verbose=True)


# In[ ]:


If we compare the above predicted rating, we can see the baseline model predicted rating as 3.62 and the optimized model predicted the rating as 3.62.


# In[ ]:


Below we are predicting rating for the same userId=4 but for a movie which this user has not interacted before i.e. movieId=3, 
by using the optimized model as shown below -


# In[36]:


# Remove _______ and complete the code
similarity_algo_optimized_user.predict(4,3, verbose=True)


# In[ ]:


If we compare the above predicted rating, we can see the baseline model predicted rating as 4 and the optimized model predicted the rating as 3.20.


# In[ ]:


Identifying similar users to a given user (nearest neighbors)
We can also find out the similar users to a given user or its nearest neighbors based on this KNNBasic algorithm. 
Below we are finding 5 most similar user to the userId=4 based on the msd distance metric


# In[37]:


similarity_algo_optimized_user.get_neighbors(4, k=5)


# In[ ]:


Implementing the recommendation algorithm based on optimized KNNBasic model
Below we will be implementing a function where the input parameters are -
data: a rating dataset
user_id: an user id against which we want the recommendations
top_n: the number of movies we want to recommend
algo: the algorithm we want to use to predict the ratings


# In[39]:


def get_recommendations(data, user_id, top_n, algo):
    
    # Creating an empty list to store the recommended movie ids
    recommendations = []
    
    # Creating an user item interactions matrix 
    user_item_interactions_matrix = data.pivot(index='userId', columns='movieId', values='rating')
    
    # Extracting those movie ids which the user_id has not interacted yet
    non_interacted_movies = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()
    
    # Looping through each of the movie id which user_id has not interacted yet
    for item_id in non_interacted_movies:
        
        # Predicting the ratings for those non interacted movie ids by this user
        est = algo.predict(user_id, item_id).est
        
        # Appending the predicted ratings
        recommendations.append((item_id, est))

    # Sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:top_n] # returing top n highest predicted rating movies for this user


# In[40]:


# Predicted top 5 movies for userId=4 with similarity based recommendation system
recommendations = get_recommendations(rating,4,5,similarity_algo_optimized_user)


# In[41]:


recommendations


# In[ ]:


Model 3: Item based Collaborative Filtering Recommendation System


# In[ ]:





# In[42]:


# Definfing similarity measure
sim_options = {'name': 'cosine',
               'user_based': False}

# Defining Nearest neighbour algorithm
algo_knn_item = KNNBasic(sim_options=sim_options,verbose=False)

# Train the algorithm on the trainset or fitting the model on train dataset 
algo_knn_item.fit(trainset)

# Predict ratings for the testset
predictions = algo_knn_item.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)


# In[ ]:


Q 4.1 What is the RMSE for baseline item based collaborative filtering recommendation system


# In[ ]:


As we can see from above, these baseline model has RMSE=1.00 on test set, we will try to improve this number later by using GridSearchCV tuning different hyperparameters of this algorithm


# In[ ]:


Q 4.2 What is the Predicted rating for an user with userId =4 and for movieId= 10 and movieId=3?


# In[43]:


algo_knn_item.predict(4,10, r_ui=4, verbose=True)


# In[ ]:


As we can see - the actual rating for this user-movie pair is 4 and predicted rating is 4.37 by this similarity based baseline model


# In[ ]:


Let's predict the rating for the same userId=4 but for a movie which this user has not interacted before i.e. movieId=3


# In[44]:


algo_knn_item.predict(4,3, verbose=True)


# In[ ]:


As we can see the estimated rating for this user-movie pair is 4.07 based on this similarity based baseline model


# In[ ]:


get_ipython().set_next_input('Q 4.3 Perform hyperparameter tuning for the baseline item based collaborative filtering recommendation system and find the RMSE for tuned item based collaborative filtering recommendation system');get_ipython().run_line_magic('pinfo', 'system')


# In[45]:


# Setting up parameter grid to tune the hyperparameters
param_grid = {'k': [20, 30,40], 'min_k': [3,6,9],
              'sim_options': {'name': ['msd', 'cosine'],
                              'user_based': [False]}
              }

# Performing 3-fold cross validation to tune the hyperparameters
grid_obj = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)

# Fitting the data
grid_obj.fit(data)

# Best RMSE score
print(grid_obj.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print(grid_obj.best_params['rmse'])


# In[ ]:


Once the grid search is complete, we can get the optimal values for each of those hyperparameters as shown above
Below we are analysing evaluation metrics - RMSE and MAE at each and every split to analyze the impact of each value of hyperparameters


# In[46]:


results_df = pd.DataFrame.from_dict(grid_obj.cv_results)
results_df.head()


# In[ ]:


Now let's build the final model by using tuned values of the hyperparameters which we received by using grid search cross-validation.


# In[47]:


# Creating an instance of KNNBasic with optimal hyperparameter values
similarity_algo_optimized_item = KNNBasic(sim_options={'name': 'msd', 'user_based': False}, k=30, min_k=6,verbose=False)

# Training the algorithm on the trainset
similarity_algo_optimized_item.fit(trainset)

# Predicting ratings for the testset
predictions = similarity_algo_optimized_item.test(testset)

# Computing RMSE on testset
accuracy.rmse(predictions)


# In[ ]:


We can see from above that after tuning hyperparameters, RMSE for testset has reduced to 0.94 from 1.00. We can say that we have been able to improve the model after hyperparameter tuning.


# In[ ]:


get_ipython().set_next_input('Q 4.4 What is the Predicted rating for an item with userId =4 and for movieId= 10 and movieId=3 using tuned item based collaborative filtering');get_ipython().run_line_magic('pinfo', 'filtering')


# In[ ]:


Let's us now predict rating for an user with userId=4 and for movieId=10 with the optimized model as shown below


# In[48]:


similarity_algo_optimized_item.predict(4,10, r_ui=4, verbose=True)


# In[ ]:


Let's predict the rating for the same userId=4 but for a movie which this user has not interacted before i.e. movieId=3, by using the optimized model:


# In[49]:


similarity_algo_optimized_item.predict(4, 3, verbose=True)


# In[ ]:


If we compare the above predicted rating, we can see the baseline model predicted rating as 4.07 and the optimized model predicted the rating as 3.86.


# In[ ]:


Identifying similar users to a given user (nearest neighbors)¶
We can also find out the similar users to a given user or its nearest neighbors based on this KNNBasic algorithm.
 Below we are finding 5 most similar user to the userId=4 based on the msd distance metric


# In[50]:


similarity_algo_optimized_item.get_neighbors(4, k=5)


# In[ ]:


Predicted top 5 movies for userId=4 with similarity based recommendation system


# In[51]:


recommendations = get_recommendations(rating, 4, 5, similarity_algo_optimized_item)


# In[52]:


recommendations


# In[ ]:


Model 4: Based Collaborative Filtering - Matrix Factorization using SVD


# In[ ]:


Build a baseline matrix factorization recommendation system


# In[53]:


# Using SVD matrix factorization
algo_svd = SVD()

# Training the algorithm on the trainset
algo_svd.fit(trainset)

# Predicting ratings for the testset
predictions = algo_svd.test(testset)

# Computing RMSE on the testset
accuracy.rmse(predictions)


# In[ ]:


We can that the baseline RMSE for matrix factorization model on testset (which is 0.89) is lower as compared to the RMSE for baseline similarity based recommendation system (which is 1.00) and it is even lesser than the RMSE for optimized similarity based recommendation system (which is 0.94)


# In[ ]:


5.2 What is the Predicted rating for an user with userId =4 and for movieId= 10 and movieId=3?


# In[ ]:


Let's us now predict rating for an user with userId=4 and for movieId=10


# In[54]:


algo_svd.predict(4, 10, r_ui=4, verbose=True)


# In[ ]:


The actual rating for this user-movie pair is 4 and predicted rating is 3.95 by this matrix factorization based baseline model.


# In[55]:


algo_svd.predict(4, 3, verbose=True)


# In[ ]:


As we can see - the actual rating for this user-item pair is 3 and predicted rating is 3.49 by this matrix factorization based baseline model. It seems like we have over estimated the rating by a small margin. We will try to fix this later by tuning the hyperparameters of the model using GridSearchCV


# In[ ]:


Improving matrix factorization based recommendation system by tuning its hyper-parameters
Q 5.3 Perform hyperparameter tuning for the baseline SVD based collaborative filtering recommendation 
get_ipython().set_next_input('system and find the RMSE for tuned SVD based collaborative filtering recommendation system');get_ipython().run_line_magic('pinfo', 'system')


# In[56]:


# Set the parameter space to tune
param_grid = {'n_epochs': [10, 20, 30], 'lr_all': [0.001, 0.005, 0.01],
              'reg_all': [0.2, 0.4, 0.6]}

# Performing 3-fold gridsearch cross validation
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)

# Fitting data
gs.fit(data)

# Best RMSE score
print(gs.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


# In[ ]:


Once the grid search is complete, we can get the optimal values for each of those hyperparameters, as shown above.
Below we are analysing evaluation metrics - RMSE and MAE at each and every split to analyze the impact of each value of hyperparameters


# In[57]:


results_df = pd.DataFrame.from_dict(gs.cv_results)
results_df.head()


# In[ ]:


Now, we will the build final model by using tuned values of the hyperparameters, which we received using grid search cross-validation above.


# In[58]:


# Building the optimized SVD model using optimal hyperparameter search
svd_algo_optimized = SVD(n_epochs=20, lr_all=0.01, reg_all=0.2)

# Training the algorithm on the trainset
svd_algo_optimized.fit(trainset)

# Predicting ratings for the testset
predictions = svd_algo_optimized.test(testset)

# Computing RMSE
accuracy.rmse(predictions)


# In[ ]:


get_ipython().set_next_input('Q 5.4 What is the Predicted rating for an user with userId =4 and for movieId= 10 and movieId=3 using SVD based collaborative filtering');get_ipython().run_line_magic('pinfo', 'filtering')
Let's us now predict rating for an user with userId=4 and for movieId=10 with the optimized model


# In[59]:


svd_algo_optimized.predict(4,10, r_ui=4, verbose=True)


# In[60]:


svd_algo_optimized.predict(4, 3, verbose=True)


# In[ ]:


get_ipython().set_next_input('Q 5.5 Predict the top 5 movies for userId=4 with SVD based recommendation system');get_ipython().run_line_magic('pinfo', 'system')


# In[61]:


get_recommendations(rating, 4, 5, svd_algo_optimized)


# In[62]:


def predict_already_interacted_ratings(data, user_id, algo):
    
    # Creating an empty list to store the recommended movie ids
    recommendations = []
    
    # Creating an user item interactions matrix 
    user_item_interactions_matrix = data.pivot(index='userId', columns='movieId', values='rating')
    
    # Extracting those movie ids which the user_id has interacted already
    interacted_movies = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].notnull()].index.tolist()
    
    # Looping through each of the movie id which user_id has interacted already
    for item_id in interacted_movies:
        
        # Extracting actual ratings
        actual_rating = user_item_interactions_matrix.loc[user_id, item_id]
        
        # Predicting the ratings for those non interacted movie ids by this user
        predicted_rating = algo.predict(user_id, item_id).est
        
        # Appending the predicted ratings
        recommendations.append((item_id, actual_rating, predicted_rating))

    # Sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return pd.DataFrame(recommendations, columns=['movieId', 'actual_rating', 'predicted_rating']) # returing top n highest predicted rating movies for this user


# In[ ]:


Here we are comparing the predicted ratings by similarity based recommendation system against actual ratings for userId=7


# In[63]:


predicted_ratings_for_interacted_movies = predict_already_interacted_ratings(rating, 7, similarity_algo_optimized_item)
df = predicted_ratings_for_interacted_movies.melt(id_vars='movieId', value_vars=['actual_rating', 'predicted_rating'])
sns.displot(data=df, x='value', hue='variable', kde=True);


# In[ ]:


Below we are comparing the predicted ratings by matrix factorization based recommendation system against actual ratings for userId=7


# In[64]:


predicted_ratings_for_interacted_movies = predict_already_interacted_ratings(rating, 7, svd_algo_optimized)
df = predicted_ratings_for_interacted_movies.melt(id_vars='movieId', value_vars=['actual_rating', 'predicted_rating'])
sns.displot(data=df, x='value', hue='variable', kde=True);


# In[65]:


# Instantiating Reader scale with expected rating scale
reader = Reader(rating_scale=(0, 5))

# Loading the rating dataset
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)

# Splitting the data into train and test dataset
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# In[ ]:


get_ipython().set_next_input('Question6: Compute the precision and recall, for each of the 6 models, at k = 5 and 10. This is 6 x 2 = 12 numerical values');get_ipython().run_line_magic('pinfo', 'values')


# In[66]:


# Function can be found on surprise documentation FAQs
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


# In[67]:


# Make list of k values
# A basic cross-validation iterator.
from surprise.model_selection import KFold
kf = KFold(n_splits=5)
K = [5, 10]

# Make list of models
models = [algo_knn_user, similarity_algo_optimized_user,algo_knn_item,similarity_algo_optimized_item, algo_svd, svd_algo_optimized]


for k in K:
    for model in models:
        print('> k={}, model={}'.format(k,model.__class__.__name__))
        p = []
        r = []
        for trainset, testset in kf.split(data):
            model.fit(trainset)
            predictions = model.test(testset, verbose=False)
            precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=3.5)

            # Precision and recall can then be averaged over all users
            p.append(sum(prec for prec in precisions.values()) / len(precisions))
            r.append(sum(rec for rec in recalls.values()) / len(recalls))
        print('-----> Precision: ', round(sum(p) / len(p), 3))
        print('-----> Recall: ', round(sum(r) / len(r), 3))


# In[ ]:


Collaborative Filtering using user-user based interaction performed well in both the k values with Precision value ~74% (k=10) and with k=5, ~74%.

Tuned SVD has better RMSE than all models but Collaborative Filtering using user-user based interaction is also giving good results based on Precsion and recall @k for K=10.

The final model will denpend on the business requirements as whether they have to minimize RMSE or go with maximizing Precision/Recall.


# In[ ]:


User-based and Item-based Collaborative Models have nearly same. User based RMSE values (0.99) while the "Item based" model's RMSE is 1.00. Clearly, tuned Collaborative Filtering Models have performed better than baseline model and the user-user based tuned model is performing better and have rmse of 0.99

The Collaborative Models use the user-item-ratings data to find similarities and make predictions rather than just predicting a random rating based on the distribution of the data. This could a reason why the Collaborative filtering performed well.

Collaborative Filtering searches for neighbors based on similarity of item (example) preferences and recommend items that those neighbors interacted while Matrix factorization works by decomposing the user-item matrix into the product of two lower dimensionality rectangular matrices.

RMSE for Matrix Factorization (0.89) is better than the Collaborative Filtering Models (~1.00).

Tuning SVD matrix factorization model is not improving the base line SVD much.

Matrix Factorization has lower RMSE due to the reason that it assumes that both items and users are present in some low dimensional space describing their properties and recommend a item based on its proximity to the user in the latent space. Implying it accounts for latent factors as well.

