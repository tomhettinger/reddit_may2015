# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:31:36 2016

@author: Tom
"""
import sqlite3

import pandas as pd
import numpy as np

import plotting


def popular_subreddits(conn):
    """Print out the top 20 subreddits by comment count, and 
    create a bar plot."""
    query = "SELECT subreddit, COUNT(subreddit) as cnt \
             FROM May2015 GROUP BY subreddit"
    subreddits = pd.read_sql_query(query, conn)
    subreddits.sort('cnt', ascending=False, inplace=True)
    top20 = subreddits.head(20)
    plotting.subreddit_bar(top20)


def count_nonzero(conn):
    for col in ['gilded', 'archived', 'downs', 'controversiality', 'edited', 'score_hidden']:
        query = "SELECT count(%s) as cnt FROM May2015 WHERE %s != 0" % (col, col)
        print col, pd.read_sql_query(query, conn)


def count_nonnull(conn):
    for col in ['distinguished', 'removal_reason', 'author_flair_text']:
        query = "SELECT count(%s) as cnt FROM May2015 WHERE %s IS NOT NULL" % (col, col)
        print col, pd.read_sql_query(query, conn)

    
def main():
    conn = sqlite3.connect("reddit.sqlite") #  54,504,410 rows total

    ### Global statistics
    #####################
    # Determine how many non-zero values are in various columns
    count_nonzero(conn)
    count_nonnull(conn)
    
    # Determine which subreddits have the most comments.
    popular_subreddits(conn)

    # Create a tiny DF with first 50 rows
    query = "SELECT * FROM May2015 LIMIT 50"
    tiny = pd.read_sql_query(query, conn)


    