# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

from sklearn.metrics import ndcg_score


def compute_NDCG(true_df: pd.DataFrame(), pred_df: pd.DataFrame())-> float:

    # Sort the true value by userId and rating. 
    # Assumption here is that the highest rating should be the more relevant movie
    true_df = true_df.sort_values(['userId','rating'], ascending = False)
    
    # Merge with predicted score 
    new_df = pd.merge(true_df, pred_df, left_on = ['userId','movieId'], right_on = ['userId','movieId'])
    true_relevance = np.asarray([new_df['rating_x']])
    scores = np.asarray([new_df['rating_y']])
    
    ndcg = ndcg_score(true_relevance, scores)
    
    return ndcg
    
    



