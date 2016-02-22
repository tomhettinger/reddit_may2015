# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:37:44 2016

@author: Tom
"""
import sqlite3

import pandas as pd
import numpy as np

import plotting


def main():
    conn = sqlite3.connect("reddit.sqlite") #  54,504,410 rows total

    ### Gilded statistics
    #####################
    # Get all gilded comments
    query = "SELECT * FROM May2015 WHERE gilded != 0"
    gilded = pd.read_sql_query(query, conn)

    # Copmare Scores (ups)
    # Grab the upvote column and downsample it to 100K rows, this takes a while
    query = "SELECT ups FROM May2015"
    ups = pd.read_sql_query(query, conn)
    ups = ups.sample(100000)
    print ups.describe()
    print gilded.ups.describe()
    # plot gilded upvote distribution
    plotting.upvote_density(gilded)
    plotting.upvote_comparison(gilded, ups.sample(17472).values)

    # How long are gilded comments?
    body_length = [len(b) for b in gilded.body.tolist()]
    print 'mean body length', np.mean(body_length)
    plotting.body_length_dist(body_length)
    query = "SELECT body FROM May2015 LIMIT 10000"
    all_body = pd.read_sql_query(query, conn)
    all_body_length = [len(b) for b in all_body.body.tolist()]
    print 'all mean body length', np.mean(all_body_length)

    # What does the distribution in subreddits look like?
    gilded.groupby('subreddit')['subreddit'].count().sort_values(ascending=False).head(20)

    # How many comments have the 'thank you' in it?
    edited_gilded = gilded.ix[gilded['edited'] > 0]
        
    c = 0
    for body in gilded.body.tolist():
        if 'thank' in body.lower():
            c += 1
    print 'gilded comments with thank yous: ', c
    print c / float(len(gilded)) * 100.0
    
    c = 0
    for body in edited_gilded.body.tolist():
        if 'thank' in body.lower():
            c += 1
    print 'edited+gilded comments with thank yous: ', c
    print c / float(len(edited_gilded)) * 100.0