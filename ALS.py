#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:07:50 2022

@author: Kirsteenng
"""

import os
import time

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType
from pyspark.mllib.recommendation import ALS

# data science imports
import math
import numpy as np
import pandas as pd

# visualization imports
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline


# spark config
spark = SparkSession \
    .builder \
    .appName("movie recommendation") \
    .config("spark.driver.maxResultSize", "96g") \
    .config("spark.driver.memory", "96g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.master", "local[12]") \
    .getOrCreate()
# get spark context
sc = spark.sparkContext # Get or instantiate a SparkContext and register it as a singleton object.
 
# path config
data_path = '/Users/Kirsteenng_1/Data Science/Movie Recommendation system'

# Load data
movies = spark.read.load(os.path.join(data_path, 'movies.csv'), format='csv', header=True, inferSchema=True)
ratings = spark.read.load(os.path.join(data_path, 'ratings.csv'), format='csv', header=True, inferSchema=True)
links = spark.read.load(os.path.join(data_path, 'links.csv'), format='csv', header=True, inferSchema=True)
tags = spark.read.load(os.path.join(data_path, 'tags.csv'), format='csv', header=True, inferSchema=True)

'''
Spark SQL and OLAP
Below are the questions I'd like to ask:

What are the ratings?
Are there any null ratings?

What is minimum number of ratings per user and minimum number of ratings per movie?
How many movies are rated by only one user?
What is the total number of users in the data sets?
What is the total number of movies in the data sets?
How many movies are rated by users? List movies not rated yet?
List all movie genres
Find out the number of movies for each category
Calculate the total rating count for every movie
Get a count plot for each rating
What are the ratings?
'''

# 1. What are the distinct ratings?
sorted(ratings.select('rating').distinct().rdd.map(lambda r: r[0]).collect())

# 2. Are there any null ratings?
ratings.filter(ratings.rating.isNull()).show()


# 3. What is minimum number of ratings per user and minimum number of ratings per movie?
min_rating_user = ratings.groupBy("userId").count().toPandas()['count'].min()
min_rating_movie = ratings.groupBy("movieId").count().toPandas()['count'].min()

# 4. How many movies are rated by only min number of users?
movie_min_user = sum(ratings.groupBy("userID").count().toPandas()['count'] == 20)
total_movies = ratings.select('movieId').distinct().count()

# 5. What is the total number of users in the data sets?
ratings.select('userId').distinct().count()

# 6. What is the total number of movies in the data sets?
ratings.select('movieId').distinct().count()

# 7. How many movies are rated by users? List movies not rated yet?
rated = ratings.select('movieId').distinct().count()
total_movie = movies.select('movieId').distinct().count()
non_rated = total_movie - rated

# Show movies that are not rated
movies.createOrReplaceTempView("movies") # need to create temporary view for both movies and ratings tables
ratings.createOrReplaceTempView("ratings")
query = 'SELECT movieId,title FROM movies WHERE movieId NOT IN \
        (SELECT DISTINCT movieId FROM ratings)'
spark.sql(query).show()

# 8. List all movie genres
movies.printSchema()
movies.select('genres').distinct().show() # this will not work because the genres are connected by |
splitter = UserDefinedFunction(lambda x: x.split('|'), ArrayType(StringType()))
movies.select(explode(splitter("genres")).alias("genres")).distinct().show()

# 9. The number of movies per genre
print('Counts of movies per genre')
movies.select('movieID', explode(splitter("genres")).alias("Genres")).groupby('Genres').count() \
    .sort(desc('count')) \
    .show()


'''
Spark ALS based approach for training model
1. Reload data
2. Split data into train, validation, test
3. ALS model selection and evaluation
4. Model testing
'''

# load data
movie_rating = sc.textFile(os.path.join(data_path, 'ratings.csv'))
# preprocess data -- only need ["userId", "movieId", "rating"]
header = movie_rating.take(1)[0]
rating_data = movie_rating \
    .filter(lambda line: line!=header) \ # operate on data that is not header
    .map(lambda line: line.split(",")) \ # split the data(userId,movieId,rating) into individual tokens
    .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) \
    .cache()
# check three rows
rating_data.take(3)

# splitting data into train, validation, test set
train, validation, test = rating_data.randomSplit([6,2,2],seed = 100)

# cache() is used to store intermediate result in memory https://sparkbyexamples.com/spark/spark-difference-between-cache-and-persist/#:~:text=Spark%20Cache%20vs%20Persist&text=Both%20caching%20and%20persisting%20are,the%20user%2Ddefined%20storage%20level.
# https://stackoverflow.com/questions/28981359/why-do-we-need-to-call-cache-or-persist-on-a-rdd
train.cache()
validation.cache()
test.cache()

# *********************** related functions for ALS ***********************
def train_ALS(train_data, validation_data, num_iters, reg_param, ranks):
    """
    Grid Search Function to select the best model based on RMSE of hold-out data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in reg_param:
            # train ALS model
            model = ALS.train(
                ratings=train_data,    # (userID, productID, rating) tuple
                iterations=num_iters,
                rank=rank,
                lambda_=reg,           # regularization param
                seed=99)
            # make prediction, recall the columns are ["userId", "movieId", "rating"]
            valid_data = validation_data.map(lambda p: (p[0], p[1]))
            predictions = model.predictAll(valid_data).map(lambda r: ((r[0], r[1]), r[2]))
            # get the rating result
            ratesAndPreds = validation_data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
            # get the RMSE
            MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
            error = math.sqrt(MSE)
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, error))
            if error < min_error:
                min_error = error
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and regularization = {}'.format(best_rank, best_regularization))
    return best_model

def plot_learning_curve(arr_iters, train_data, validation_data, reg, rank):
    """
    Plot function to show learning curve of ALS
    """
    errors = []
    for num_iters in arr_iters:
        # train ALS model
        model = ALS.train(
            ratings=train_data,    # (userID, productID, rating) tuple
            iterations=num_iters,
            rank=rank,
            lambda_=reg,           # regularization param
            seed=100)
        # make prediction
        valid_data = validation_data.map(lambda p: (p[0], p[1]))
        predictions = model.predictAll(valid_data).map(lambda r: ((r[0], r[1]), r[2]))
        # get the rating result
        ratesAndPreds = validation_data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        # get the RMSE
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        error = math.sqrt(MSE)
        # add to errors
        errors.append(error)

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(arr_iters, errors)
    plt.xlabel('number of iterations')
    plt.ylabel('RMSE')
    plt.title('ALS Learning Curve')
    plt.grid(True)
    plt.show()

def get_movieId(df_movies, fav_movie_list):
    """
    return all movieId(s) of user's favorite movies
    
    Parameters
    ----------
    df_movies: spark Dataframe, movies data
    
    fav_movie_list: list, user's list of favorite movies
    
    Return
    ------
    movieId_list: list of movieId(s)
    """
    movieId_list = []
    for movie in fav_movie_list:
        movieIds = df_movies \
            .filter(movies.title.like('%{}%'.format(movie))) \
            .select('movieId') \
            .rdd \
            .map(lambda r: r[0]) \
            .collect()
        movieId_list.extend(movieIds)
    result = list(set(movieId_list))
    #print(result)
    return result

def get_inference_data(train_data, df_movies, movieId_list):
    """
    return a rdd with the userid and all movies (except ones in movieId_list)

    Parameters
    ----------
    train_data: spark RDD, ratings data

    df_movies: spark Dataframe, movies data
    
    movieId_list: list, list of movieId(s)

    Return
    ------
    inference data: Spark RDD
    """
    # get new user id
    new_id = train_data.map(lambda r: r[0]).max() + 1
    # return inference rdd
    return df_movies.rdd \
        .map(lambda r: r[0]) \
        .distinct() \
        .filter(lambda x: x not in movieId_list) \
        .map(lambda x: (new_id, x))
        
def add_new_user_to_data(train_data, movieId_list, spark_context):
    """
    add new rows with new user, user's movie and ratings to
    existing train data

    Parameters
    ----------
    train_data: spark RDD, ratings data
    
    movieId_list: list, list of movieId(s)

    spark_context: Spark Context object
    
    Return
    ------
    new train data with the new user's rows
    """
    # get new user id
    new_id = train_data.map(lambda r: r[0]).max() + 1
    # get max rating
    max_rating = train_data.map(lambda r: r[2]).max()
    # create new user rdd
    user_rows = [(new_id, movieId, max_rating) for movieId in movieId_list]
    new_rdd = spark_context.parallelize(user_rows)
    
    # return new train data
    return train_data.union(new_rdd)


def make_recommendation(best_model_params, ratings_data, df_movies,fav_movie_list, n_recommendations, spark_context):
    """
    return top n movie recommendation based on user's input list of favorite movies


    Parameters
    ----------
    best_model_params: dict, {'iterations': iter, 'rank': rank, 'lambda_': reg}

    ratings_data: spark RDD, ratings data

    df_movies: spark Dataframe, movies data

    fav_movie_list: list, user's list of favorite movies

    n_recommendations: int, top n recommendations

    spark_context: Spark Context object

    Return
    ------
    list of top n movie recommendations
    """
    # modify train data by adding new user's rows
    movieId_list = get_movieId(df_movies, fav_movie_list)

    train_data = add_new_user_to_data(ratings_data, movieId_list, spark_context) 
    #wouldnt this cause duplication if the user's data is already in the training dataset?
    
    # train best ALS
    model = ALS.train(
        ratings=train_data,
        iterations=best_model_params.get('iterations', None),
        rank=best_model_params.get('rank', None),
        lambda_=best_model_params.get('lambda_', None),
        seed=100)
    
    # get inference rdd
    inference_rdd = get_inference_data(ratings_data, df_movies, movieId_list)
    
    # inference
    predictions = model.predictAll(inference_rdd).map(lambda r: (r[1], r[2]))
    
    # get top n movieId
    topn_rows = predictions.sortBy(lambda r: r[1], ascending=False).take(n_recommendations)
    topn_ids = [r[0] for r in topn_rows]
    
    # return movie titles
    return df_movies.filter(movies.movieId.isin(topn_ids)) \
                    .select('title') \
                    .rdd \
                    .map(lambda r: r[0]) \
                    .collect()
# ********************************************************************

# hyper-param config
num_iterations = 10
ranks = [8, 10, 12, 14, 16, 18, 20]
reg_params = [0.001, 0.01, 0.05, 0.1, 0.2]

# grid search and select best model
start_time = time.time()
final_model = train_ALS(train, validation, num_iterations, reg_params, ranks)

print ('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))

# create an array of num_iters
iter_array = list(range(1, 11))
# create learning curve plot
plot_learning_curve(iter_array, train, validation, 0.05, 20)

# use final model for predictions
test_data = test.map(lambda x: (x[0],x[1]))
predictions = final_model.predictAll(test_data).map(lambda x:((x[0], x[1]), x[2]))
ratesAndPreds = test.map(lambda t:((t[0], t[1]), t[2])).join(predictions)

# get the mean square error
MSE = ratesAndPreds.map(lambda e: (e[1][0] - e[1][1]) ** 2).mean()
error = math.sqrt(MSE)
print('The MSE on test data = ',round(error,4))



# my favorite movies
my_favorite_movies = ['Spider Man','Iron Man','Thor']

# get recommends
recommends = make_recommendation(
    best_model_params={'iterations': 10, 'rank': 15, 'lambda_': 0.05}, 
    ratings_data=rating_data, 
    df_movies=movies, 
    fav_movie_list=my_favorite_movies, 
    n_recommendations=5, 
    spark_context=sc)

print('Recommendations for {}:'.format(my_favorite_movies[0]))
for i, title in enumerate(recommends):
    print('{0}: {1}'.format(i+1, title))
    
