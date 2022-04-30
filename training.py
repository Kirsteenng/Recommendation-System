'''
Contains functions to train ALS model.

'''

# spark imports
from pyspark.mllib.recommendation import ALS

# libraries for plotting and analysis
import config as cf
from matplotlib import pyplot as plt
import math

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
            print("Shape of validation= ", valid_data.count())
            predictions = model.predictAll(valid_data).map(lambda r: ((r[0], r[1]), r[2]))
            print("Shape of predictions= ", predictions.count())
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

def get_movieId(df_movies, fav_movie_list,movies):
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

def get_inference_data(train_data, df_movies, movieId_list,user_id):
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
    
    # return inference rdd
    return df_movies.rdd \
        .map(lambda r: r[0]) \
        .distinct() \
        .filter(lambda x: x not in movieId_list) \
        .map(lambda x: (user_id, x))
        
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
    # get new user id, assume input is by new user
    new_id = train_data.map(lambda r: r[0]).max() + 1 
    # get max rating, assign max rating to favourite movie
    max_rating = train_data.map(lambda r: r[2]).max()
    # create new user rdd
    user_rows = [(new_id, movieId, max_rating) for movieId in movieId_list]
    new_rdd = spark_context.parallelize(user_rows)
    
    # return new train data
    return train_data.union(new_rdd)


def make_recommendation(movies,best_model_params, ratings_data, df_movies,fav_movie_list, n_recommendations, spark_context,user_id):
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
    movieId_list = get_movieId(df_movies, fav_movie_list,movies)

    train_data = add_new_user_to_data(ratings_data, movieId_list, spark_context) 
    #wouldnt this cause duplication if the user's data is already in the training dataset?
    new_id = train_data.map(lambda r: r[0]).max()
    
    # train best ALS
    model = ALS.train(
        ratings=train_data,
        iterations=best_model_params.get('iterations', None),
        rank=best_model_params.get('rank', None),
        lambda_=best_model_params.get('lambda_', None),
        seed=100)   
    
    # get inference rdd
    inference_rdd = get_inference_data(ratings_data, df_movies, movieId_list,new_id)
    
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
                    
def make_predictions(movies,input_movie_list,user_id,training_data, model,n_recommendations):
    input_movie_id = get_movieId(movies, input_movie_list, movies) #
    input_movie_inference = get_inference_data(training_data,movies,input_movie_id,user_id)
    predictions = model.predictAll(input_movie_inference).map(lambda r: (r[1], r[2]))
    print("Number of row in prediction model= ", predictions.count())
    topn_rows = predictions.sortBy(lambda r: r[1], ascending=False).take(n_recommendations)
    topn_ids = [r[0] for r in topn_rows]
    print(topn_rows)
    
    return movies.filter(movies.movieId.isin(topn_ids)).select('title').rdd.map(lambda r : r[0]).collect()

def get_rating_data(ratings, header):
    rating_data = ratings.filter(lambda line: line!=header)  \
        .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) \
        .cache()
    return rating_data
        

