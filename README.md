# Movie Recommendation System using Alternating Least Square (ALS) Matrix Factorization

* Dataset: [Movielens Dataset](https://grouplens.org/datasets/movielens/latest/)  
In this exercise I have downloaded the [small dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip). Create a folder called `/data/` and save the files from the zip under the `/data/` folder. The default pathway is set for `/data/`, but you can pass a different pathway as an argument (e.g. `python main.py -data_path <pathway>`)

* Requirements
Download the Python requirements to your local machine using the requirements.txt file.  e.g. `pip install -r requirements.txt`

Goal: 
1. Understand the concept of Matrix Factorization and Alternating Least Square(ALS). This is achieved by implementing ALS according to this [paper](http://yifanhu.net/PUB/cf.pdf)
2. Understand how to set up, configure and use Spark ML. 
3. Create a ALS recommendation system that can take in user details, such as movie history, personal characteristics, as input and output movie recommendation.

Procedure:
1. Conduct EDA to understand the dataset. The EDA procedures and answers are listed under EDA.py.  
2. Build and train ALS model, which is under training.py.
3. Test model.
4. Evaluate ALS model using NDCG.


Project Owner: 
* Kirsteenng

Contributors:
* Kirsteenng
* sambk17

Credit and reference:
* https://github.com/yeomko22/ALS_implementation/blob/master/als.ipynb
* https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1
* http://yifanhu.net/PUB/cf.pdf
* https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1
