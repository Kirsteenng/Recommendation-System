# Movie Recommendation System using Alternating Least Square (ALS) Matrix Factorization

Dataset: [Movielens Dataset](https://grouplens.org/datasets/movielens/latest/)  
In this exercise I have downloaded the [small dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip). Save the pathname under the variable data_path in config.py. 

Goal: 
1. Understand the concept of Matrix Factorization and Alternating Least Square(ALS). This is achieved by implementing ALS according to this [paper](http://yifanhu.net/PUB/cf.pdf)
2. Understand how to set up, configure and use Spark ML. 
3. Create a ALS recommendation system that can take in user details, such as movie history, personal characteristics, as input and output movie recommendation.

Procedure:
1. Conduct EDA to understand the dataset. The EDA procedures and answers are listed under EDA.py.  
2. Build and train ALS model, which is under training.py.
3. Test model.
4. Evaluate ALS model using RMSE. Other evaluation methods such as nDCG@k and MAP@k will also be compared.


Project Owner: 
* Kirsteenng

Contributors:
* Kirsteenng
* sambk17

Credit:
* https://github.com/yeomko22/ALS_implementation/blob/master/als.ipynb
* https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1
