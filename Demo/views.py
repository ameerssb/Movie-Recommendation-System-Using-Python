from django.shortcuts import render
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import Movie_Recommendation_Notebook as MD
movieList = MD.movie_list
genreList = MD.new_list
no_of_users = MD.ratings['userId'].unique().tolist()

top_rec = 10
# Create your views here.

def home(request):
    return render(request, 'Home.html')

def MoviesList(request):
    listof = movieList.values.tolist()
    #print(listof)
    context = {'listof': listof}
    return render(request, 'movieslist.html', context)

def Genres(request):
    m_name = request.GET.get('genre')
    df = MD.best_movies_by_genre(m_name,top_rec)
    listof = df.iloc[:, 0].tolist() 
    #print(listof)
    context = {'listof': listof, 'genreList': genreList, 'name': m_name, 'NoOf' : top_rec}
    return render(request, 'genres.html', context)

def WatchBy(request):
    m_name = request.GET.get('title')
    df = MD.get_other_movies(m_name, top_rec)
    listo = df['perc_who_watched'].tolist()
    res = []
    index = df.index
    a_list = list(index)
    for x in range(top_rec):
        res.append([index[x],listo[x]])
    listof = res
    #print(listof)
    context = {'listof': listof, 'List': movieList['title'], 'name': m_name, 'NoOf' : top_rec}    
    return render(request, 'WatchBy.html', context)

def SimilarMovies(request):
    m_name = request.GET.get('title')
    mv = MD.movie_list.query("title == '%s' " % m_name)
    movie_id = mv['movieId'].iloc[0]
    df = MD.print_similar_movies(movie_id,top_rec)
    listof = df
    context = {'listof': listof, 'List': movieList['title'], 'name': m_name, 'NoOf' : top_rec}
    return render(request, 'similar.html', context)

def SimilarContent(request):
    m_name = request.GET.get('title')
    movie_id = MD.indicies[m_name]
    df = MD.get_similar_movies_based_on_content(movie_id,top_rec)
    listof = df.values.tolist()
    #print(listof)
    context = {'listof': listof, 'List': movieList['title'], 'name': m_name, 'NoOf' : top_rec}
    return render(request, 'content.html', context)

def UserRec(request):
    idss = request.GET.get('userid')
    ids = int(idss)
    df = MD.get_similar_movies_based_on_user(ids,top_rec)
    listof = df.values.tolist()
    #print(listof)
    context = {'listof': listof, 'List': no_of_users, 'id': ids, 'NoOf' : top_rec}
    return render(request, 'userrec.html', context)
