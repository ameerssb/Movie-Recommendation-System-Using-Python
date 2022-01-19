# Movie-Recommendation-System-using-python

## Data Set 
This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

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
1. python 3
2. Scipy
3. Numpy
4. Pandas
5. sk-learn

