# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:36:20 2016

@author: Tom
"""
import sqlite3

import pandas as pd
import numpy as np

import plotting


def get_depth(df, id_number):
    """Find the depth of a comment by looking at the chain of
    parent comments. Dataframe must be indexed by id, and
    must contain a 'depth' column (even if all NaN)."""
    # Check if this id is in the dataframe at all
    if id_number not in df.index:
        return 999
        
    # Check if this depth has already been determined
    thisDepth = df.ix[id_number, 'depth']
    if not np.isnan(thisDepth):
        return thisDepth
    
    # Otherwise, check if parent is comment or link posting
    parent = df.ix[id_number, 'parent_id']
    parent_type, parent_id = parent.split('_')
    if parent_type == 't3':
        # parent is the posting
        df.ix[id_number, 'depth'] = 0
    elif parent_type == 't1':
        # parent is another comment
        df.ix[id_number, 'depth'] = 1 + get_depth(df, parent_id)
    else:
        raise Exception('unkown parent type' + parent_type)

    return df.ix[id_number, 'depth']


def convert_score(df):
    """Conver score to score_alt, by adding a constant = -MIN + 1,
    and taking the log."""
    df['score_alt'] = df['score']
    min_score = df.score_alt.min()
    if min_score < 1:
        offset = abs(min_score) + 1
    else:
        offset = 0
    df.score_alt += offset
    df.score_alt = np.log10(df.score_alt)


def plot_mean_score_vs_depth(df, age=True):
    mean_scores = []
    for i in range(31):
        if age:
            theseScores = df.ix[df.age_round == i, 'score']
        else:
            theseScores = df.ix[df.depth == i, 'score']
        mean_scores.append((i, theseScores.mean(), theseScores.std(), theseScores.quantile(.025), theseScores.quantile(.975)))
    plotting.avg_score_vs_depth(mean_scores, age=age)
    return mean_scores


def make_word_cloud(df, start, stop):
    low, hi = pd.to_datetime([start, stop]).astype(np.int64) // 10**9
    week = df.ix[(df.created_utc >= low) & (df.created_utc < hi) & (df.score > 50)]
    plotting.word_cloud(week)
    

def main():    
    # Get all comments for the "worldnews" subreddit
    conn = sqlite3.connect("reddit.sqlite") #  54,504,410 rows total
    query = "SELECT * FROM May2015 WHERE subreddit='worldnews'"
    worldnews = pd.read_sql_query(query, conn)
    worldnews.set_index('id', inplace=True)

    # Create features
    worldnews['age'] = (worldnews.retrieved_on - worldnews.created_utc) / 3600. / 24.
    worldnews['age_round'] = worldnews['age'].round()
    #plotting.score_vs_age(worldnews.sample(20000))
    plotting.age_density(worldnews)
    plot_mean_score_vs_depth(worldnews, age=True)
   
    # plot upvote density
    plotting.upvote_density(worldnews)
    
    # Calculate the depth of each comment.  This takes a 10s of minutes
    worldnews['depth'] = np.nan
    for i, idx in enumerate(worldnews.index):
        if i % 10000 == 0:
            print i
        get_depth(worldnews, idx)
    worldnews.depth.to_csv('worldnews_depth.csv')
    # Set depths with incomplete ancestry to nan
    worldnews.ix[worldnews.depth > 900, 'depth'] = np.nan
    
    # Plot depth hist and score vs depth
    plotting.depth_density(worldnews)
    plotting.score_vs_depth(worldnews, log=False)
    plotting.score_depth_box_plot(worldnews)
    
    # Look at avg score as function of depth
    plot_mean_score_vs_depth(worldnews, age=False)
    
    # Word clouds by week
    make_word_cloud(worldnews, '2015-04-27', '2015-05-03')
    make_word_cloud(worldnews, '2015-05-04', '2015-05-10')
    make_word_cloud(worldnews, '2015-05-11', '2015-05-17')
    make_word_cloud(worldnews, '2015-05-18', '2015-05-24')
    make_word_cloud(worldnews, '2015-05-25', '2015-05-31')
    
    # Word clouds individual days
    make_word_cloud(worldnews, '2015-05-04', '2015-05-05')
    make_word_cloud(worldnews, '2015-05-13', '2015-05-14')
    make_word_cloud(worldnews, '2015-05-18', '2015-05-19')
    make_word_cloud(worldnews, '2015-05-29', '2015-05-30')
    
    print worldnews.depth.corr(worldnews.score)
    print worldnews.depth.corr(worldnews.score_alt)
    print worldnews.age.corr(worldnews.score)
    print worldnews.age.corr(worldnews.score_alt)
    print worldnews.ix[worldnews.depth <= 10, 'depth'].corr(worldnews.score)
    