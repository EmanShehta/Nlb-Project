import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import numpy as np
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

movie = pd.read_excel('movies.xlsx')
rating = pd.read_excel('ratings.xlsx')

def movieProcessing(movie):
    #Extract year (Year) from title
    movie['year'] = movie.title.str.extract('(\(\d\d\d\d\))', expand=False)
    #Extract year from (year)
    movie['year'] = movie.year.str.extract('(\d\d\d\d)', expand=False)
    #remove (year) from title
    movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '')
    #to remove ending whitespace
    movie['title'] = movie['title'].apply(lambda x: x.strip())
    # convert genres to list
    movie['genres'] = movie.genres.str.split('|')
    # to save in memory
    movie.movieId = movie.movieId.astype('int32')
    #fill missing values
    movie.year.fillna(0, inplace=True)
    #convert Year to low size
    movie.year = movie.year.astype('int16')

    return movie
    # movies_df_new_mem = movie.memory_usage()

# result = movie.merge(rating,on="movieId",how = "left")
# df =pd.read_excel('Results.xlsx')
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
#Removing the parentheses
# df['year'] = df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
# df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared remove any ending white space
# df['title'] = df['title'].apply(lambda x: x.strip())
# df['genres'] = df.genres.str.split('|')
# df.movieId = df.movieId.astype('int32')
# fill missing values
# df.year.fillna(0, inplace=True)
# df.userId.fillna(0, inplace=True)
# df.rating.fillna(0, inplace=True)
# df.timestamp.fillna(0, inplace=True)
# df.year = df.year.astype('int16')
# CopyData = df.copy(deep=True)

movie = movieProcessing(movie);
CopyData = movie.copy(deep=True)

for index, row in movie.iterrows():
    for genre in row['genres']:
        CopyData.at[index, genre] = 1

# df = df.fillna(0)
CopyData=CopyData.fillna(0)
# end of movie processing

# df = df.drop('timestamp',axis=1,inplace=True)

def RatingPreProcessing(rating):
    rating.drop('timestamp', axis=1, inplace=True)


RatingPreProcessing(rating)


Result = pd.merge(rating,movie)


# so on a scale of 0 to 5, with 0 min and 5 max, see Lawrence's movie ratings below
# Lawrence_movie_ratings = [
#             {'title':'Predator', 'rating':4.9},
#             {'title':'Final Destination', 'rating':4.9},
#             {'title':'Mission Impossible', 'rating':4},
#             {'title':"Beverly Hills Cop", 'rating':3},
#             {'title':'Exorcist, The', 'rating':4.8},
#             {'title':'Waiting to Exhale', 'rating':3.9},
#             {'title':'Avengers, The', 'rating':4.5},
#             {'title':'Omen, The', 'rating':5.0}
#          ]
# Lawrence_movie_ratings = pd.DataFrame(Lawrence_movie_ratings)

Result = Result[Result['userId'].isin([1])]

Result.drop(['movieId','userId','genres','year'],axis=1,inplace=True)

def  Recommend(Lawrence_movie_ratings):
    # Extracting movie Ids from movies_df and updating lawrence_movie_ratings with movie Ids.
    Lawrence_movie_Id = movie[movie['title'].isin(Lawrence_movie_ratings['title'])]
    # Merging Lawrence movie Id and ratings into the lawrence_movie_ratings data frame.
    # This action implicitly merges both data frames by the title column.
    Lawrence_movie_ratings = pd.merge(Lawrence_movie_Id, Lawrence_movie_ratings)
    # Display the merged and updated data frame.
    Lawrence_movie_ratings
    #Dropping information we don't need such as year and genres
    Lawrence_movie_ratings = Lawrence_movie_ratings.drop(['genres','year'], 1)
    #Final input dataframe
    #If a movie you added in above isn't here, then it might not be in the original
    #dataframe or it might spelled differently, please check capitalisation.
    Lawrence_movie_ratings
    # filter the selection by outputing movies that exist in both lawrence_movie_ratings and movies_with_genres
    Lawrence_genres_df = CopyData[CopyData.movieId.isin(Lawrence_movie_ratings.movieId)]
    Lawrence_genres_df
    # First, let's reset index to default and drop the existing index.
    Lawrence_genres_df.reset_index(drop=True, inplace=True)
    # Next, let's drop redundant columns
    Lawrence_genres_df.drop(['movieId','title','genres','year'], axis=1, inplace=True)
    # Let's view chamges
    Lawrence_genres_df
    # Let's find the dot product of transpose of Lawrence_genres_df by lawrence rating column
    predictedshape =Lawrence_movie_ratings.shape[0]
    matrixshape=Lawrence_genres_df.shape[0]
    if predictedshape>matrixshape:
        Lawrence_movie_ratings= Lawrence_movie_ratings.iloc[:matrixshape , :]
    else:
        Lawrence_genres_df = Lawrence_genres_df.iloc[:predictedshape, :]

    # print('Shape of Lawrence_movie_ratings is:', Lawrence_movie_ratings.shape)
    # print('Shape of Lawrence_genres_df is:', Lawrence_genres_df.shape)

    Lawrence_profile = Lawrence_genres_df.T.dot(Lawrence_movie_ratings.rating)

    # let's set the index to the movieId
    movies_with_genres = CopyData.set_index(CopyData.movieId)

    # Deleting four unnecessary columns.
    movies_with_genres.drop(['movieId','title','genres','year'], axis=1, inplace=True)

    # Multiply the genres by the weights and then take the weighted average.
    recommendation_table_df = (movies_with_genres.dot(Lawrence_profile)) / Lawrence_profile.sum()

    # Let's sort values from great to small
    recommendation_table_df.sort_values(ascending=False, inplace=True)

    # first we make a copy of the original movies_df
    copy = movie.copy(deep=True)
    # Then we set its index to movieId
    copy = copy.set_index('movieId', drop=True)
    # Next we enlist the top 20 recommended movieIds we defined above
    top_20_index = recommendation_table_df.index[:10].tolist()
    # finally we slice these indices from the copied movies df and save in a variable
    recommended_movies = copy.loc[top_20_index, :]
    # Now we can display the top 20 movies in descending order of preference
    print(recommended_movies)

Recommend(Result)