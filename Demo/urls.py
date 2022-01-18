from . import views
from django.urls import path

urlpatterns = [
    path('', views.home, name='Home'),
    path('moviesList', views.MoviesList, name='moviesList'),
    path('Genres', views.Genres, name='Genres'),
    path('watchBy', views.WatchBy, name='WatchBy'),
    path('similarMovies', views.SimilarMovies, name='SimilarMovies'),
    path('similarContent', views.SimilarContent, name='SimilarContent'),
    path('UserRecommendation', views.UserRec, name='UserRecommendation'),
]