'''
Explore and understand dataset through predefined questions.
'''
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType

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
def EDA(spark, ratings, movies, links, tags):
# 1. What are the distinct ratings?
    print('These are the distinct ratings: ')
    print(sorted(ratings.select('rating').distinct().rdd.map(lambda r: r[0]).collect()))
    
    # 2. Are there any null ratings?
    ratings.filter(ratings.rating.isNull()).show()
    
    
    # 3. What is minimum number of ratings per user and minimum number of ratings per movie?
    min_rating_user = ratings.groupBy("userId").count().toPandas()['count'].min()
    print('The minimum number of ratings per user =',min_rating_user)
    min_rating_movie = ratings.groupBy("movieId").count().toPandas()['count'].min()
    print('The minimum number of ratings per movie =', min_rating_movie)
    
    # 4. How many movies are rated by only min number of users?
    movie_min_user = sum(ratings.groupBy("userID").count().toPandas()['count'] == 20)
    total_movies = ratings.select('movieId').distinct().count()
    print('Total number of movies rated by %d users = %d' % (min_rating_user, total_movies))
    
    # 5. What is the total number of users in the data sets?
    print('Total number of users = ',ratings.select('userId').distinct().count())
    
    # 6. What is the total number of movies in the data sets?
    print('Total number of movies = ',ratings.select('movieId').distinct().count())
    
    # 7. How many movies are rated by users? List movies not rated yet?
    rated = ratings.select('movieId').distinct().count()
    total_movie = movies.select('movieId').distinct().count()
    non_rated = total_movie - rated
    print('Total movies rated by users = ', rated)
    print('Total movies not rated by users = ', non_rated)
    
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
    print('The followings are all the movie genres in this database: ')
    movies.select(explode(splitter("genres")).alias("genres")).distinct().show()
    
    # 9. The number of movies per genre
    print('Counts of movies per genre')
    movies.select('movieID', explode(splitter("genres")).alias("Genres")).groupby('Genres').count() \
        .sort(desc('count')) \
        .show()


