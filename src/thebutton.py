# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:37:01 2016

@author: Tom
"""
import sqlite3

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix

import plotting


def author_analysis(df):
    """Find the number of comment posts for each unique author in 
    the data set."""
    print 'unique authors: ', len(df.author.unique())
    author_comment_counts = df.groupby('author').author.count().sort_values()
    author_comment_counts.name = 'author_comment_count'
    print author_comment_counts[-20:]
    plotting.author_comment_dist(author_comment_counts.tolist()[:-1])
    plotting.author_comment_cumsum(author_comment_counts.tolist()[:-1])    
    return author_comment_counts


def click_time_analysis(df):
    """Plot distribution of button click times, based on flair text.
    Ignores cheaters."""
    df.flair_time = df.author_flair_text
    idx = (df.author_flair_css_class != 'cheater') & \
          (df.author_flair_css_class != 'cant-press') & \
          (df.author_flair_css_class != 'no-press') & \
          (df.author_flair_css_class.notnull())
    print df.ix[idx, 'author_flair_text'].unique()
    df.flair_time[~idx] = None
    df.flair_time = df.flair_time.map(lambda x: np.nan if x is None else int(x[:-1]))
    plotting.click_time_dist(df)    


def css_class_analysis(df):
    """ Plot the distribution of flair awarded using the css_class field."""
    print df.groupby('author_flair_css_class')['author_flair_css_class'].count()
    plotting.flair_distribution(df)


def impute_null_with_string(df):
    """Replace all missing values in flair_text and flair_css_class with
    the string 'na'."""
    print 'author_flair_text == Null: ', df.author_flair_text.isnull().sum()
    df.author_flair_text = df.author_flair_text.fillna('na')
    print 'author_flair_css_class == Null', df.author_flair_css_class.isnull().sum()
    df.author_flair_css_class = df.author_flair_css_class.fillna('na')



def get_most_recent_comments(df):
    """Return a dataframe with only the most recent comment of each
    unique author. Keeps all comments for [deleted] authors."""
    author_dict = {}
    deleted_idx = []
    for row in df.iterrows():
        rowID, thisFrame = row
        if thisFrame.author == '[deleted]':
            deleted_idx.append(rowID)
            continue
            
        if thisFrame.author not in author_dict:
            author_dict[thisFrame.author] = (rowID, thisFrame.author, thisFrame.created_utc)
        elif thisFrame.created_utc > author_dict[thisFrame.author][2]:
            author_dict[thisFrame.author] = (rowID, thisFrame.author, thisFrame.created_utc)
    print '%d unique, non-deleted authors.' % len(author_dict)
    print '%d deleted author comments.' % len(deleted_idx)
    idx = [tup[0] for tup in author_dict.values()]
    idx.extend(deleted_idx)
    return df.ix[idx]
    

def drop_unnecessary(df):
    print 'removal_reason', df.removal_reason.notnull().sum()
    print 'archived', (df.archived != 0).sum()
    print 'downs', (df.downs != 0).sum()
    print 'ups != score', (df.ups != df.score).sum()
    print 'score_hidden', (df.score_hidden != 0).sum()
    print 'distinguished', df.distinguished.notnull().sum()
    print 'edited', (df.edited != 0).sum()
    print 'controversiality', (df.controversiality != 0).sum()
    print 'gilded', (df.gilded != 0).sum()    
    print 'subreddit_id', df.subreddit_id.unique()
    print 'subreddit_id', df.subreddit_id.isnull().sum()    
    df.drop(['ups', 'downs', 'archived', 'removal_reason', 'score_hidden', 'subreddit_id'],
            axis=1, inplace=True)


def model_pressers_gboost(X_train, X_test, y_train, y_test):
    # Dimensionality reduction
    svd = TruncatedSVD(n_components=500)
    svd.fit(X_train)
    print svd.explained_variance_ratio_.sum()
    reduced_X_train = svd.transform(X_train)
    reduced_X_test = svd.transform(X_test)
    
    # Train the model    
    clf = GradientBoostingClassifier(n_estimators=50, subsample=0.75, max_depth=3)
    clf.fit(reduced_X_train, y_train)

    # Check the validity
    pred = clf.predict(reduced_X_train)
    print "Accuracy on train set: ", 100*accuracy_score(pred, y_train)
    pred = clf.predict(reduced_X_test)
    print "Accuracy on validation: ", 100*accuracy_score(pred, y_test)
    
    # ROC
    y_scores = clf.predict_proba(reduced_X_test)[:, 1]
    print "ROC AUC = ", roc_auc_score(y_test, y_scores)
    fpr, tpr, __ = roc_curve(y_test, y_scores)
    plotting.roc(fpr, tpr)


def model_color_gboost(X_train, X_test, y_train, y_test):
    # Train the model
    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, subsample=0.80, max_depth=4)
    clf.fit(tfidf_train, y_train)

    # Check the validity
    pred = clf.predict(tfidf_train.toarray())
    print "Accuracy on train set: ", 100*accuracy_score(pred, y_train)
    pred = clf.predict(tfidf_test.toarray())
    print "Accuracy on validation: ", 100*accuracy_score(pred, y_test)
    print confusion_matrix(y_test, pred, 
                           labels=['press-6', 'press-5', 'press-4', 'press-3', 'press-2', 'press-1'])




def model_pressers_logistic(X_train, X_test, y_train, y_test, feature_names):
    # Train the model
    clf = LogisticRegression() # play with C=1.0 to alter regularization
    clf.fit(X_train, y_train)

    # Check the validity
    pred = clf.predict(X_train)
    print "Accuracy on train set: ", 100*accuracy_score(pred, y_train)
    pred = clf.predict(X_test)
    print "Accuracy on validation: ", 100*accuracy_score(pred, y_test)
    
    # ROC
    y_scores = clf.predict_proba(X_test)[:, 1]
    print "ROC AUC = ", roc_auc_score(y_test, y_scores)
    fpr, tpr, __ = roc_curve(y_test, y_scores)
    plotting.roc(fpr, tpr)

    # Important features
    feature_weights = clf.coef_[0]
    feature_tup = zip(feature_weights, feature_names)
    feature_tup.sort(key=lambda x: abs(x[0]))
    return feature_tup[-30:]
    
    
def model_color_logistic(X_train, X_test, y_train, y_test, feature_names):
    clf = LogisticRegression(C=0.1, multi_class='multinomial', solver='newton-cg')
    clf.fit(X_train, y_train)

    # Check the validity
    pred = clf.predict(X_train)
    print "Accuracy on train set: ", 100*accuracy_score(pred, y_train)
    pred = clf.predict(X_test)
    print "Accuracy on validation: ", 100*accuracy_score(pred, y_test)
    print confusion_matrix(y_test, pred, 
                           labels=['press-6', 'press-5', 'press-4', 'press-3', 'press-2', 'press-1'])
    
    # Important features
    feature_weights = clf.coef_[0]
    feature_tup = zip(feature_weights, feature_names)
    feature_tup.sort(key=lambda x: abs(x[0]))
    return feature_tup[-30:]



def main():
    conn = sqlite3.connect("reddit.sqlite") #  54,504,410 rows total
    query = "SELECT * FROM May2015 WHERE subreddit='thebutton'"  # 135,670 rows
    thebutton = pd.read_sql_query(query, conn)

    # Author analysis
    author_comment_counts = author_analysis(thebutton)
    thebutton = pd.merge(thebutton, pd.DataFrame(author_comment_counts), how='left', left_on='author', right_index=True)

    # Get the most recent post from each unique author
    unique_author_frame = get_most_recent_comments(thebutton)
   
    # Look at distribution of click times
    click_time_analysis(thebutton)
    click_time_analysis(unique_author_frame)
   
    # Replace null values of flair and class with 'na'
    impute_null_with_string(thebutton)
    impute_null_with_string(unique_author_frame)
    
    # Look at the distribution of css class
    css_class_analysis(thebutton)
    css_class_analysis(unique_author_frame)

    
    ### Predicting flair colors

    # Drop uncessary columns
    drop_unnecessary(thebutton)
    # Drop rows with no target (flair css == 'na')
    thebutton = thebutton[thebutton.author_flair_css_class != 'na']
    # Remove authors with more than 10 comments (but keep [deleted])
    thebutton = thebutton.ix[thebutton.author_comment_count <= 10]
    print thebutton.groupby(thebutton.author_flair_css_class).author_flair_css_class.count()

    ## MODEL 1
    # For the first model, classify comments into either pressed or hasn't pressed
    thebutton['has_pressed'] = thebutton.author_flair_css_class.map(lambda x: x[:-1] == 'press-')
    # ... and ignore 'cheaters' and 'cant-press'
    valid_player_idx = thebutton.author_flair_css_class.map(lambda x: x not in ['cheater', 'cant-press'])
    model_01_frame = thebutton[valid_player_idx]    
    
    # Split into training and validation
    corpus = model_01_frame.body.tolist()
    labels = model_01_frame.has_pressed.values
    X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.25, random_state=42)
                                   
    # Vectorize the body text
    vec = TfidfVectorizer(min_df=5, ngram_range=[1,2], stop_words='english') # consider increasing min_df
    vec.fit(X_train)
    print len(vec.vocabulary_)
    feature_names = vec.get_feature_names()
    tfidf_train = vec.transform(X_train)
    tfidf_test = vec.transform(X_test)

    # Fit models
    imp = model_pressers_logistic(tfidf_train, tfidf_test, y_train, y_test, feature_names)
    print imp
    model_pressers_gboost(tfidf_train, tfidf_test, y_train, y_test)
    
    
    ## MODEL 2
    # For the second model, classify comments by color, considering only those that were pressed.
    model_02_frame = thebutton[thebutton.has_pressed == True]
    
    # Split into training and validation
    corpus = model_02_frame.body.tolist()
    labels = model_02_frame.author_flair_css_class.values
    X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.25, random_state=1337)
    
    # Vectorize the body text
    vec = TfidfVectorizer(min_df=5, ngram_range=[1,2], stop_words='english') # consider increasing min_df
    vec.fit(X_train)
    print len(vec.vocabulary_)
    feature_names = vec.get_feature_names()
    tfidf_train = vec.transform(X_train)
    tfidf_test = vec.transform(X_test)
    
    # Fit models
    imp = model_color_logistic(tfidf_train, tfidf_test, y_train, y_test, feature_names)
    model_color_gboost(tfidf_train, tfidf_test, y_train, y_test)

    