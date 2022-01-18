#!/usr/bin/env python
# coding: utf-8


#importing the required libraries
import numpy as np
import pandas as pd
import pickle
#import matrix_factorization_utilities
import scipy.sparse as sp
from scipy.sparse.linalg import svds


# Reading the ratings data
ratings = pd.read_csv('Dataset/ml-latest-small/ratings.csv')


# In[3]:


len(ratings)


# In[4]:


#Just taking the required columns
ratings = ratings[['userId', 'movieId','rating']]


# In[5]:


# Checking if the user has rated the same movie twice, in that case we just take max of them
ratings_df = ratings.groupby(['userId','movieId']).aggregate(np.max)


# In[6]:


# In this case there are no such cases where the user has rated the same movie twice.
len(ratings_df)


# In[7]:


# Inspecting the data
#ratings.head()


# In[8]:


#ratings_df.head()


# In[9]:


# Counting no of unique users
#len(ratings['userId'].unique())


# In[10]:


#Getting the percentage count of each rating value 
count_ratings = ratings.groupby('rating').count()
count_ratings['perc_total']=round(count_ratings['userId']*100/count_ratings['userId'].sum(),1)


# In[11]:


#count_ratings


# In[12]:


#Visualising the percentage total for each rating
#count_ratings['perc_total'].plot.bar()


# In[13]:


#reading the movies dataset
movie_list = pd.read_csv('Dataset/ml-latest-small/movies.csv')


# In[14]:


#len(movie_list)


# In[15]:


# insepcting the movie list dataframe
#movie_list.head()


# In[16]:


# reading the tags datast
tags = pd.read_csv('Dataset/ml-latest-small/tags.csv')


# In[17]:


# inspecting the tags data frame
#tags.head()


# In[18]:


# inspecting various genres
genres = movie_list['genres']


# In[19]:


#genres.head()


# In[20]:


genre_list = ""
for index,row in movie_list.iterrows():
        genre_list += row.genres + "|"
#split the string into a list of values
genre_list_split = genre_list.split('|')
#de-duplicate values
new_list = list(set(genre_list_split))
#remove the value that is blank
new_list.remove('')
#inspect list of genres
#new_list


# In[21]:


#Enriching the movies dataset by adding the various genres columns.
movies_with_genres = movie_list.copy()
#index_df = ['movieId']
#movies_with_genres.set_index(index_df, drop=True, inplace=True, verify_integrity=True)
#movies_with_genres.index.name = None
for genre in new_list :
    movies_with_genres[genre] = movies_with_genres.apply(lambda _:int(genre in _.genres), axis = 1)


# In[22]:


#movies_with_genres.head()


# In[23]:


#Calculating the sparsity
no_of_users = len(ratings['userId'].unique())
no_of_movies = len(ratings['movieId'].unique())

sparsity = round(1.0 - len(ratings)/(1.0*(no_of_movies*no_of_users)),3)
#print(sparsity)


# In[24]:


# Counting the number of unique movies in the dataset.
#len(ratings['movieId'].unique())


# In[35]:


# Finding the average rating for movie and the number of ratings for each movie
avg_movie_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean','count']))
avg_movie_rating.index.name = None
avg_movie_rating['movieId']= avg_movie_rating.index


# In[36]:


# inspecting the average movie rating data frame
#avg_movie_rating.head()


# In[37]:


#len(avg_movie_rating)


# In[38]:


#calculate the percentile count. It gives the no of ratings at least 70% of the movies have
np.percentile(avg_movie_rating['count'],70)


# In[39]:


#Get the average movie rating across all movies 
avg_rating_all=ratings['rating'].mean()
avg_rating_all
#set a minimum threshold for number of reviews that the movie has to have
min_reviews=30
#min_reviews
movie_score = avg_movie_rating.loc[avg_movie_rating['count']>min_reviews]
#movie_score.head()


# In[40]:


#len(movie_score)


# In[41]:


#create a function for weighted rating score based off count of reviews
def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['count']
    R = x['mean']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[42]:


#movie_score


# In[43]:


#movies_with_genres


# In[44]:


#join movie details to movie ratings
movie_score = pd.merge(movie_score,movies_with_genres,on='movieId')
#join movie links to movie ratings
#movie_score = pd.merge(movie_score,links,on='movieId')
#movie_score.head()


# In[45]:


#Calculating the weighted score for each movie
movie_score['weighted_score'] = movie_score.apply(weighted_rating, axis=1)
#movie_score.head()


# In[46]:


#list top scored movies over the whole range of movies
pd.DataFrame(movie_score.sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score','genres']][:10])


# In[47]:


# Gives the best movies according to genre based on weighted score which is calculated using IMDB formula
def best_movies_by_genre(genre,top_n):
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title']][:top_n])


# In[48]:


#run function to return top recommended movies by genre
#best_movies_by_genre('Musical',10)  


# In[49]:


#run function to return top recommended movies by genre
#best_movies_by_genre('Action',10)  


# In[50]:


#run function to return top recommended movies by genre
#best_movies_by_genre('Children',10)  


# In[51]:


#run function to return top recommended movies by genre
#best_movies_by_genre('Drama',10)  


# In[52]:


# Creating a data frame that has user ratings accross all movies in form of matrix used in matrix factorisation
ratings_df = pd.pivot_table(ratings, index='userId', columns='movieId', aggfunc=np.max)


# In[53]:


#ratings_df.head()


# In[57]:


#ratings_df_for_matrix = ratings_df.to_numpy()


# In[58]:



# Apply low rank matrix factorization to find the latent features
#U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df_for_matrix,
#                                                                    num_features=5,
#                                                                    regularization_amount=1.0)


# In[59]:


#ratings_df


# In[60]:


#merging ratings and movies dataframes
ratings_movies = pd.merge(ratings,movie_list, on = 'movieId')


# In[61]:


#ratings_movies.head()


# In[62]:


#ratings_movies


# In[63]:


def get_other_movies(movie_name,nm):
    #get all users who watched a specific movie
    df_movie_users_series = ratings_movies.loc[ratings_movies['title']==movie_name]['userId']
    #convert to a data frame
    df_movie_users = pd.DataFrame(df_movie_users_series,columns=['userId'])
    #print(df_movie_users.head())
    #get a list of all other movies watched by these users
    other_movies = pd.merge(df_movie_users,ratings_movies,on='userId')
    #print(other_movies.head())    
    #get a list of the most commonly watched movies by these other user
    other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId',ascending=False)
    #print(other_users_watched.head())
    other_users_watched['perc_who_watched'] = round(other_users_watched['userId']*100/other_users_watched['userId'][0],1)
    #print(other_users_watched.head())
    return other_users_watched[:nm]


# In[64]:


# Getting other top 10 movies which are watched by the people who saw 'Gone Girl'
#get_other_movies('Gone Girl (2014)')


# In[65]:


from sklearn.neighbors import NearestNeighbors


# In[66]:


#avg_movie_rating.head()


# In[67]:


#only include movies with more than 10 ratings
movie_plus_10_ratings = avg_movie_rating.loc[avg_movie_rating['count']>=10]
#print(len(movie_plus_10_ratings))


# In[68]:


#movie_plus_10_ratings


# In[69]:


filtered_ratings = pd.merge(movie_plus_10_ratings, ratings, on="movieId")
#len(filtered_ratings)


# In[70]:


#filtered_ratings.head()


# In[71]:


#create a matrix table with movieIds on the rows and userIds in the columns.
#replace NAN values with 0
movie_wide = filtered_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
#movie_wide.head()


# In[72]:


#specify model parameters
model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
#fit model to the data set
model_knn.fit(movie_wide)


# In[74]:


# In[75]:

#Getting the movies list with only genres like Musical and other such columns
movie_content_df_temp = movies_with_genres.copy()
movie_content_df_temp.set_index('movieId')
movie_content_df = movie_content_df_temp.drop(columns = ['movieId','title','genres'])
movie_content_df = movie_content_df.to_numpy()
#movie_content_df


#create a series of the movie id and title
indicies = pd.Series(movie_content_df_temp.index, movie_content_df_temp['title'])
#indicies 


#Gets the top 10 nearest neighbours got the movie
def print_similar_movies(query_index,top) :
    #get the list of user ratings for a specific userId
    query_index_movie_ratings = movie_wide.loc[query_index,:].values.reshape(1,-1)
    #get the closest 10 movies and their distances from the movie specified
    distances,indices = model_knn.kneighbors(query_index_movie_ratings,n_neighbors = top + 1) 
    #write a lopp that prints the similar movies for a specified movie.
    listall = []
    for i in range(0,len(distances.flatten())):
        #get the title of the random movie that was chosen
        get_movie = movie_list.loc[movie_list['movieId']==query_index]['title']
        #for the first movie in the list i.e closest print the title
        if i==0:
            print('Recommendations for {0}:\n'.format(get_movie))
        else :
            #get the indiciees for the closest movies
            indices_flat = indices.flatten()[i]
            #get the title of the movie
            get_movie = movie_list.loc[movie_list['movieId']==movie_wide.iloc[indices_flat,:].name]['title']
#    similar_movies = pd.DataFrame(movie_content_df_temp[['title','genres']].iloc[movie_indices])            
            listall.append(get_movie)
            #print the movie
            #print('{1}'.format(get_movie))
#    sani = pd.DataFrame(listall)
    return listall

# In[76]:


#print_similar_movies(112552)


# In[77]:


#print_similar_movies(1)


# In[78]:


#print_similar_movies(96079)


# In[79]:


#movies_with_genres.head()


# In[80]:



# In[81]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(movie_content_df,movie_content_df)


# In[82]:


# Similarity of the movies based on the content
#cosine_sim


# In[83]:



# In[84]:


def get_similar_movies_based_on_content(movie_index,top) :
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:top]
#    print(sim_scores)
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    #print(movie_indices)
    similar_movies = pd.DataFrame(movie_content_df_temp[['title','genres']].iloc[movie_indices])
    return similar_movies


# In[85]:


#indicies["Skyfall (2012)"]


# In[87]:


#get_similar_movies_based_on_content(7955)


# In[88]:


#get ordered list of movieIds
item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))),columns=['movieId'])
#add in data frame index value to data frame
item_indices['movie_index']=item_indices.index
#inspect data frame
#item_indices.head()


# In[89]:


#get ordered list of movieIds
user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))),columns=['userId'])
#add in data frame index value to data frame
user_indices['user_index']=user_indices.index
#inspect data frame
#user_indices.head()


# In[90]:


#join the movie indices
df_with_index = pd.merge(ratings,item_indices,on='movieId')
#join the user indices
df_with_index=pd.merge(df_with_index,user_indices,on='userId')
#inspec the data frame
#df_with_index.head()


# In[91]:


#import train_test_split module
from sklearn.model_selection import train_test_split
#take 80% as the training set and 20% as the test set
df_train, df_test= train_test_split(df_with_index,test_size=0.2)
#print(len(df_train))
#print(len(df_test))


# In[92]:


#df_train.head()


# In[93]:


#df_test.head()


# In[94]:


n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]
#print(n_users)
#print(n_items)


# In[95]:


#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_train.itertuples():
    #set the value in the column and row to 
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    train_data_matrix[line[5], line[4]] = line[3]
train_data_matrix.shape


# In[96]:


#Create two user-item matrices, one for training and another for testing
test_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_test[:1].itertuples():
    #set the value in the column and row to 
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    #print(line[2])
    test_data_matrix[line[5], line[4]] = line[3]
    #train_data_matrix[line['movieId'], line['userId']] = line['rating']
#test_data_matrix.shape


# In[97]:


#pd.DataFrame(train_data_matrix).head()


# In[98]:


#df_train['rating'].max()


# In[99]:


from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    #select prediction values that are non-zero and flatten into 1 array
    prediction = prediction[ground_truth.nonzero()].flatten() 
    #select test values that are non-zero and flatten into 1 array
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    #return RMSE between values
    return sqrt(mean_squared_error(prediction, ground_truth))


# In[100]:


#Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
for i in [1,2,5,20,40,60,100,200]:
    #apply svd to the test data
    u,s,vt = svds(train_data_matrix,k=i)
    #get diagonal matrix
    s_diag_matrix=np.diag(s)
    #predict x with dot product of u s_diag and vt
    X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
    #calculate rmse score of matrix factorisation predictions
    rmse_score = rmse(X_pred,test_data_matrix)
    rmse_list.append(rmse_score)
    #print("Matrix Factorisation with " + str(i) +" latent features has a RMSE of " + str(rmse_score))


# In[101]:


#Convert predictions to a DataFrame
mf_pred = pd.DataFrame(X_pred)
#mf_pred.head()


# In[102]:


df_names = pd.merge(ratings,movie_list,on='movieId')
#df_names.head()


# In[105]:


#gets recommendation based on users by choosing a user ID
def get_similar_movies_based_on_user(user_id,top):
    #get movies rated by this user id
    users_movies = df_names.loc[df_names["userId"]==user_id]
    user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
    #get movie ratings predicted for this user and sort by highest rating prediction
    sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
    #rename the columns
    sorted_user_predictions.columns=['ratings']
    #save the index values as movie id
    sorted_user_predictions['movieId']=sorted_user_predictions.index
    #print("Top 10 predictions for User " + str(user_id))
    #display the top 10 predictions for this user
    pred=pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:top]
    return pred[['title','genres']]


# In[107]:


#count number of unique users
numUsers = df_train.userId.unique().shape[0]
#count number of unitque movies
numMovies = df_train.movieId.unique().shape[0]
#print(len(df_train))
#print(numUsers) 
#print(numMovies) 


# In[108]:


#Separate out the values of the df_train data set into separate variables
Users = df_train['userId'].values
Movies = df_train['movieId'].values
Ratings = df_train['rating'].values
#print(Users),print(len(Users))
#print(Movies),print(len(Movies))
#print(Ratings),print(len(Ratings))


# In[ ]:




