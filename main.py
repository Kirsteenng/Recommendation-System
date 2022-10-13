'''
Allows users to input desired movies and output movie recommendations.
'''

import os
import time
import math
import argparse

# spark imports
from pyspark.sql import SparkSession
from training import train_ALS, plot_learning_curve, make_recommendation, make_predictions
from EDA import EDA
from NDCG import compute_NDCG

def main(args):
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
    movies = spark.read.load(os.path.join(args.data_path, 'movies.csv'), format='csv', header=True, inferSchema=True)
    ratings = spark.read.load(os.path.join(args.data_path, 'ratings.csv'), format='csv', header=True, inferSchema=True)
    links = spark.read.load(os.path.join(args.data_path, 'links.csv'), format='csv', header=True, inferSchema=True)
    tags = spark.read.load(os.path.join(args.data_path, 'tags.csv'), format='csv', header=True, inferSchema=True)


    # ******************

    # Conduct EDA
    EDA(spark, ratings, movies, links, tags)


    # Train ALS model
    movie_rating = sc.textFile(os.path.join(args.data_path, 'ratings.csv'))
    header = movie_rating.take(1)[0]
    rating_data = movie_rating.filter(lambda line: line!=header).map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))).cache()
        
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
    final_model = train_ALS(train, validation, num_iterations, reg_params, ranks)  # a matrix that contains all user and product with ratings.

    print ('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))


    # create an array of num_iters
    iter_array = list(range(1, 11))
    # create learning curve plot
    plot_learning_curve(iter_array, train, validation, 0.05, 20)



    # insert evaluation stage using NDCG

    test_x = test.map(lambda p: (p[0], p[1]))
    test_y = test.map(lambda p:p[2])

    pred_test = final_model.predictAll(test_x).map(lambda r: (r[0],r[1],r[2]))
    pred_test_df = pred_test.map(lambda x:(x[0],x[1],x[2])).toDF(('userId','movieId','rating')).toPandas()



    # find top ten movies for each user for original and predicted
    test_df = test.map(lambda x:(x[0],x[1],x[2])).toDF(('userId','movieId','rating')).toPandas()
    ndcg = compute_NDCG(test_df,pred_test_df)
    print('NDCG score: ',ndcg)


    # =============================================================================
    # # Note that the input length and predictied output length are not the same
    # # Question: why is it not the same?
    # # Script to inspect the difference
    # test_x.count()
    # pred_test.count()89
    # test_tuple = list(zip(test_df.userId, test_df.movieId))
    # pred_tuple = list(zip(pred_test_df.userId, pred_test_df.movieId))
    # diff = set(test_tuple) - set(pred_tuple)
    # gp = test_df.groupby('movieId').agg('count').reset_index()
    # =============================================================================

    # User's favorite movies
    my_user_id = 360 #this user id allows final_model to get relevant data for the same user
    my_favorite_movies = ['Batman','Fantasia','Scream'] # turn this into user ID, product ID and find similar movies

    # make recommendations using the final model
    recommendations = make_predictions(movies = movies,
                    input_movie_list = my_favorite_movies,
                    user_id = my_user_id,
                    training_data = rating_data,
                    n_recommendations= 10,
                    model = final_model)


    print('Recommendations for {}:'.format(my_favorite_movies[0]))
    for i, title in enumerate(recommendations):
        print('{0}: {1}'.format(i+1, title))

    # get recommendation by assuming user is new ie iserId = max(current user id)+1) 
    recommends = make_recommendation(
        movies = movies,
        best_model_params={'iterations': 10, 'rank': 15, 'lambda_': 0.05}, 
        ratings_data=rating_data, 
        df_movies=movies, 
        fav_movie_list=my_favorite_movies, 
        n_recommendations=5, 
        spark_context=sc,user_id = my_user_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/")

    args = parser.parse_args()

    main(args)