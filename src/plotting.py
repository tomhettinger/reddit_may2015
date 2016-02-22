# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:30:19 2016

@author: Tom
"""
from wordcloud import WordCloud, STOPWORDS
STOPWORDS.update(['deleted', 'thing', 'still', 'things', 'lot', 'gt', 'reddit', 'really', 'something', 'https', 'also', 'many', 'even', 'much', 'will'])
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
ORANGERED = '#ff4500'
PERIWINKLE = '#5f99cf'


def word_cloud(df):
    text = ' '.join(df.body.tolist())
    wordcloud = WordCloud(max_font_size=40, stopwords=STOPWORDS, relative_scaling=.5).generate(text)
    plt.figure(figsize=[12, 8])
    plt.imshow(wordcloud)
    plt.axis("off")
    #print wordcloud.words_


def subreddit_bar(df):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    ind = np.arange(len(df))
    ax.bar(ind + 0.1, df.cnt, width=0.8, color='dodgerblue')
    ax.set_xticks(ind + 0.5)
    ax.set_xticklabels(df.subreddit, rotation=70)
    ax.set_title("Subreddit Popularity")
    ax.set_ylabel("Number of Comments")
    

def avg_score_vs_depth(depth_mean_scores, age=False):
    depth, avg_score, std, q1, q3 = zip(*depth_mean_scores)
    yerr_low = np.array(avg_score) - np.array(q1)
    yerr_hi = np.array(q3) - np.array(avg_score)
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    #ax.plot(depth, avg_score, yerr=err, color=PERIWINKLE)
    ax.errorbar(depth, avg_score, yerr=[yerr_low, yerr_hi], color=PERIWINKLE, ls='-', marker='o')
    if age:
        ax.set_xlabel('Age of Comment (days)')
        ax.set_xlim(0, 35)
    else:
        ax.set_xlabel('Depth of Comment')
        ax.set_xlim(-1, 20)
    ax.set_ylabel(r'Mean Comment Score (bars indicate 95% of sample)')
    ax.set_title("/r/worldnews May 2015")
    ax.set_ylim(-20, 50)
    
    
def score_depth_box_plot(df):
    plt.figure(figsize=[12, 6])
    ax = sns.boxplot(df.depth, df.score, color=PERIWINKLE)
    ax.set_ylim(-5, 10)
    ticks = np.arange(0, 80, 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_title("/r/worldnews May 2015")

   
def score_vs_depth(df, log=False):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    ax.plot(df.depth, df.score, color=PERIWINKLE, marker='.', ls='none', alpha=0.3)
    ax.set_title("/r/worldnews May 2015")
    ax.set_ylabel("Score (positive only)")
    ax.set_xlabel("Comment Depth")
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.7, 1E2)
        ax.set_ylim(0.7, 1E4)
    else:
        ax.set_xlim(-1, 20)
        ax.set_ylim(-20, 20)

def score_vs_age(df):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    ax.plot(df.age, df.score, color=PERIWINKLE, marker='.', ls='none', alpha=0.3)
    ax.set_title("/r/worldnews May 2015")
    ax.set_ylabel("Score (positive only)")
    ax.set_xlabel("Comment Age (days)")
    ax.set_yscale('log')
    ax.set_xlim(0, 30)
    ax.set_ylim(0.7, 1E4)


def depth_density(df):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    n, edges, __ = ax.hist(df.depth, color=PERIWINKLE, bins=80, range=[-0.5, 79.5])
    print edges
    ax.set_yscale('log')
    ax.set_xlim(-5, 80)
    ax.set_ylim(0.7)
    ax.set_ylabel('Number of Comments')
    ax.set_xlabel('Comment Depth')
    ax.set_title('/r/worldnews May 2015')


def age_density(df):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    n, edges, __ = ax.hist(df.age, color=PERIWINKLE, bins=60, range=[0,60])
    print edges
    ax.set_yscale('log')
    ax.set_xlim(0, 60)
    ax.set_ylim(0.7)
    ax.set_ylabel('Number of Comments')
    ax.set_xlabel('Comment Age (days)')
    ax.set_title('/r/worldnews May 2015')


def upvote_density(df):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    n, edges, __ = ax.hist(df.ups, color=ORANGERED, bins=70, range=[-1000, 6000])
    ax.set_yscale('log')
    ax.set_ylabel('Number of Comments')
    ax.set_xlabel('Score')

    
def upvote_comparison(gilded, allups):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    n, edges, __ = ax.hist(gilded.ups, color='gold', bins=70, range=[-1000, 6000], label='Gilded', alpha=0.8)
    n, edges, __ = ax.hist(allups, color=ORANGERED, bins=70, range=[-1000, 6000], label='All', alpha=0.6)
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel('Number of Comments')
    ax.set_xlabel('Score')
    
    
def body_length_dist(gilded_body):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    n, edges, __ = ax.hist(gilded_body, bins=40, color='gold')
    ax.legend()    
    ax.set_yscale('log')
    ax.set_ylabel('Number of Comments')
    ax.set_xlabel('Length of Comment Body')
    
    
def flair_distribution(df):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    ax = sns.countplot(x='author_flair_css_class', data=df.ix[df.author != '[deleted]'], 
             order=['no-press', 'press-6', 'press-5', 'press-4', 'press-3', 'press-2', 'press-1', 'cant-press', 'cheater', 'na'],
             palette=['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'white', 'brown', 'brown'])
    ax.set_title('/r/thebutton May 2015')
    
    
def click_time_dist(df):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    ax.axvline(11, color='red', ls='--')
    ax.axvline(21, color=ORANGERED, ls='--')
    ax.axvline(31, color='goldenrod', ls='--')
    ax.axvline(41, color='green', ls='--')
    ax.axvline(51, color='blue', ls='--')
    ax.axvline(60, color='purple', ls='--')
    n, edges, __ = ax.hist(df.flair_time.dropna().values, range=[-0.5, 60.5], bins=61, color=ORANGERED, zorder=100)
    #print edges
    ax.set_xlim(62, -2)
    ax.set_xlabel('Flair Time')
    ax.set_ylabel('Number of Comments')
    ax.set_title('/r/thebutton May 2015')
    

def author_comment_dist(comment_counts):
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    n, edges, __ = ax.hist(comment_counts, color=ORANGERED, bins=48, range=[0,1200])#, range=[0, 1400], bins=24)
    ax.set_yscale('log')
    ax.set_ylim(0.7, 1E5)
    ax.set_xlabel('Number of Comments')
    ax.set_ylabel('Number of Authors')
    ax.set_title('/r/thebutton May 2015')
    
    
def author_comment_cumsum(comment_counts):
    x = np.array(comment_counts)
    x = sorted(x)
    y = np.cumsum(x)
    fig = plt.figure(figsize=[12,6])
    ax = fig.gca()
    ax.plot(x, y, marker='.', ls='none')
    ax.set_xlabel('Comments per Individual Author')
    ax.set_ylabel('Total Comments  (authors having < x comments)')
    ax.set_title('/r/thebutton May 2015')
   
   
def roc(fpr, tpr):
    fig = plt.figure(figsize=[8,8])
    ax = fig.gca()
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr, tpr, color=ORANGERED)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')