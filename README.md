# Movie-Recommendation-System-using-python

## Data Set 
The data set used for this notebook is the 1M ratings data set from MovieLens. This contains 1M ratings of movies from 7120 movies and 14,025 Users. This data set includes:

* **movieId**
* **userId**
* **rating**

In addition a data set of the movies includes the movie name and genres.
* **movieId**
* **title**
* **genres**

## Demo:
### you can view the demo using python manage.py runserver

 
## Approaches Used :
### new user recommendation based on genre
This type of advice is straightforward but really valuable. Because they help consumers overcome the difficulty of a cold start. That is, we can make recommendations to the user without knowing anything about them. We can move to some of the more complicated models, which are detailed below, after collecting some reviews from the user or getting some extra information about the user.

In the notebook, formula given by **IMDB** was used to calculate the best movies according to various genres and they can be recommended to any new user.

## Most Commonly watched movie by people who watched thesame movie
This recommender takes the approach of looking at at all users who have watched a particular movie and then counts the returns the most popular movie returned by that group.

## Finding similar movies based on ratings
Here just based on the ratings of the users for different movies, we use K nearest neighbours algorithm to find the movies which are similar.

### based on content
Here we just information about the movies, in this case the information of genres to predict the most similar movies.



## Required Tools
2. Scipy
3. Numpy
4. Pandas
5. sk-learn
6. python 3

