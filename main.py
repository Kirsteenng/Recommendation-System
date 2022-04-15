'''
Allows users to input desired movies and output movie recommendations.
'''

import os
import time

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType
from pyspark.mllib.recommendation import ALS
import config # this contains the data path
from training import *
from EDA import EDA


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
 

# Load data
movies = spark.read.load(os.path.join(data_path, 'movies.csv'), format='csv', header=True, inferSchema=True)
ratings = spark.read.load(os.path.join(data_path, 'ratings.csv'), format='csv', header=True, inferSchema=True)
links = spark.read.load(os.path.join(data_path, 'links.csv'), format='csv', header=True, inferSchema=True)
tags = spark.read.load(os.path.join(data_path, 'tags.csv'), format='csv', header=True, inferSchema=True)


# ******************

# Conduct EDA
EDA(spark, ratings, movies, links, tags)


# Train ALS model
movie_rating = sc.textFile(os.path.join(data_path, 'ratings.csv'))
header = movie_rating.take(1)[0]
rating_data = movie_rating.filter(lambda line: line!=header)  \
    .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) \
    .cache()
    
train, validation, test = rating_data.randomSplit([6,2,2],seed = 100)
train.cache()
validation.cache()
test.cache()


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

# User's favorite movies
my_favorite_movies = ['Spider Man','Iron Man','Thor']

# get recommendation
recommends = make_recommendation(
    movies = movies,
    best_model_params={'iterations': 10, 'rank': 15, 'lambda_': 0.05}, 
    ratings_data=rating_data, 
    df_movies=movies, 
    fav_movie_list=my_favorite_movies, 
    n_recommendations=5, 
    spark_context=sc)

print('Recommendations for {}:'.format(my_favorite_movies[0]))
for i, title in enumerate(recommends):
    print('{0}: {1}'.format(i+1, title))
    