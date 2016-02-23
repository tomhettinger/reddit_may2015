# Reddit Comments May 2015

This is a short exploratory analysis of comments posted on the [www.reddit.com](www.reddit.com) website during the month of May 2015.  The data can be retrieved here from [www.kaggle.com](https://www.kaggle.com/reddit/reddit-comments-may-2015).



## Site-wide exploration

Which subreddits had the most comments in May 2015?

```SQL
SELECT subreddit, COUNT(subreddit) AS cnt
FROM May2015 GROUP BY subreddit;
```

<center>
![subreddit_popularity](figures/subreddit_popularity.png)
</center>

Of the 50,138 subreddits, the top 20 most commented subreddits account for 14,688,182 / 54,504,410 = 27% of all comments. 

Some other interesting statistics:
* 33.9% of comments had author flair
* 0.8% of comments were authored by ‘distinguished’ individuals (special, admin, moderator, etc.)
* 3.1% of comments had been edited
* 0.03% of comments were gilded with ‘reddit gold’
* 28.7% of comments that had been gilded, were also edited.



## Gilded comments

As mentioned above, 0.03% of comments were gilded.  28.7% of all gilded comments had edits.  The prevalence of edits are attributed, in part, by thank you’s, which are found in 17% of all gilded comments and 50% of all edited, gilded comments.  For example,

    “You can't leave Mouse out!  
    Edit: gold?  Awesome thank you!”


    “CHICAGO PIZZA > NY PIZZA  
    edit: thanks for the gold kind stranger!”

The distribution of scores is significantly different for the gilded comments than the entire set.  Using randomly selected comments from the entire data set, the distribution of all scores compared to gilded looks like:

<center>
![gilded_scores](figures/gilded_all_upvote_comparison.png)
</center>

with (mean, standard deviation) = (536, 1062) and (6, 48) for gilded comments and all comments respectively.  The increased numbers of highly negative scores indicates people are rewarding controversial content as well.

In terms of subreddits, the most gilded subreddits are common with those subreddits with the most comments overall.  Eight of the most gilded subreddits do not appear in the top commented, including:

> /r/promos, /r/gifs, /r/baseball, /r/IAmA, /r/thebutton, /r/tifu, /r/gaming, /r/politics.

Whereas the following subreddits are under-gilded:

> /r/pcmasterrace, /r/DestinyTheGame, /r/soccer, /r/DotA2, /r/GlobalOffensive, /r/hockey, /r/movies, /r/SquaredCircle.  

The mean length of a gilded comment’s body is 629 characters (4.5 times the length of a tweet on Twitter).  For comparison, the mean length of the comment bodies from the first 10,000 rows in the dataset (not pure random) is 147.  Longer comments have a higher gilding rate than shorter comments.

<center>
![body_length](figures/gilded_body_length.png)
</center>

The data set has a total size of 30GB.  For more in-depth analysis, I’ve reduced the working data set size by investigating individual subreddits.




## /r/thebutton

[The Button](www.reddit.com/r/thebutton) was a meta-game and social experiment hosted by Reddit that featured an online button and countdown timer that would reset each time the button was pressed. The experiment was hosted on the social networking website Reddit beginning on April 1 and was active until June 5, 2015. [https://en.wikipedia.org/wiki/The_Button_(Reddit)](https://en.wikipedia.org/wiki/The_Button_(Reddit))

Users were only allowed a single press of the button, and only users registered before April 1 could participate. User flair was awarded to pressers according to how many seconds were left on the timer when they pressed the button.

| Time Clicked      | Color  |
| ------------      | -----  |
| 60s-52s           | Purple |
| 51s-42s           | Blue   |     
| 41s-32s           | Green  |    
| 31s-22s           | Yellow |  
| 21s-12s           | Orange | 
| 11s-0s            | Red    |
| Did not click     | Gray   |
| Not able to click | White  |
| Cheater           | Purple |

As authors have my comments, I’ve selected the most recent comment of each unique author to look at flair distributions.  Of the 135,670 comments, there are 43,618 unique authors plus 10,875 occurrences of [deleted] where the author is unknown.  Ignoring the [deleted] accounts, the distribution of comments per author:

<center>
![author_comment_counts](figures/author_comment_counts.png)
</center>

and the cumulative total number of comments, as a function of increasing author activity:

<center>
![author_cumsum](figures/author_cumsum.png)
</center>

Returning to the flair, the distribution of flair awarded is illustrated here.  I’m using only the most recent comment from any author, and ignoring ‘[deleted]’ authors as they do not have flair.

<center>
![flair_dist](figures/flair_dist_unique_author.png)
</center>

Next, I’ve plotted the distribution of click times as reported in the flair (ignoring values for ‘cheaters’).

<center>
![click_times](figures/click_time_dist_unique_author.png)
</center>

There are peaks at around when a new flair color first becomes available.  There is a strong signal at t=0s and t=1s, where users are trying to keep the timer alive.  There is also a strong peak at t=42s.  This particular number must [have some meaning](https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#Answer_to_the_Ultimate_Question_of_Life.2C_the_Universe.2C_and_Everything_.2842.29).  By far though, the most common click time was t=60s among vocal reddit users.  


### Predicting Flair Color

If the flairs of each user were turned off, could we predict what flair they were given, based on the body of the comment?  We’ll build a model to determine if we can.  This model assumes that the body of a comment has correlations with the flair assigned to a user.  

We’ll clean the data by removing comments with missing values for flair (including [deleted] authors).  Also, we only keep comments from users with <10 total comments.  These cuts yield a total of 73,427 comments.

For our first model, we look only at comments with flair == ‘no-press’ and ‘press-X’.  We will try to predict whether someone is a presser or a non-presser.  The corpus of comment body’s were TF-IDF vectorized with a min_df=5, including bigrams, and excluding stop words.  I fit a logistic regression on a training set which yielded an accuracy score on the validation set of 0.630, and an ROC AUC of 0.670.

<center>
![roc_curve](figures/presser_lr_roc.png)
</center>

The features with the strongest coefficients include (from strongest to less strong):

    filthy non, grey, shade, towel, waited, orange, day presser, regrets, 
    pure, 60s, remain, yellow, wanted, tempted, lucky, team60s, presser, 
    just pressed, clicked, signed, felt, stay, green, pressers, regret, 
    temptation, filthy pressers, red, lurking, strong

<center>
![thebutton_wordcloud](figures/thebutton_wordcloud.png)
</center>

I also tried fitting a gradient boosting classifier to the training set.  I reduced the dimensionality of the features from over 9000 to 500 using TruncatedSVD.  I then ran the classifier with a learning rate = 0.1, max depth = 3, subsampling = 0.75, and 50 iterations.  The resulting accuracy and AUC are 0.627 and 0.650, respectively.  The logistic model outperformed the gradient boosting classifier (although there was not extensive tuning of gb parameters).

We now look at a model that attempts to recover the flair color of a comment.  Here, only comments with flair_css_class == ‘press-?’ are considered.  With a logistic regression, the accuracy is 0.43, with the following confusion matrix:

|         | press-6 | press-5 | press-4 | press-3 | press-2 | press-1 |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| press-6 | 2823    |    10   |     3   |     4   |     9   |   660   |
| press-5 | 670     |   36    |    4    |    1    |    5    |  196    |
| press-4 | 427     |    3    |  50     |    4    |    1    |  137    |
| press-3 | 505     |    0    |    3    |   57    |    9    |  169    |
| press-2 | 666     |    4    |    2    |    0    |   94    |  280    |
| press-1 | 841     |    2    |    4    |    3    |    7    | 1192    |

Using a gradient boosting model, the number of True Positives increases for all classes, but decreases for the Press-1 class.  The accuracy remains small at 0.435.


# /r/worldnews

I briefly took a look at /r/worldnews.  I engineered two new features from the data. First, I calculated the age of a comment at the time of data retrieval (retrieved_on – created_utc).  The age of the comments may affect the number of sub-comments or score.  The distribution of ages is illustrated here.

<center>
![post_age](figures/worldnews_age.png)
</center>

The second feature is comment depth.  Using the parent_id, for each comment I was able to trace how many parents comments lived between the given comment and the original thread post.  Top-level comments were given depth=0.  The distribution of comment depth is

<center>
![depth_dist](figures/worldnews_depth.png)
</center>

I wanted to see if either of these features are related to the comment score.  The distribution of comment scores is shown below, with a median score of 1, and the first and third quartiles at 1 and 3.  Indeed, 90% of comment scores rest between -4 and 20.

<center>
![worldnews_scores](figures/worldnews_scores.png)
</center>

As seen below there is no correlation with score and comment age (r = -0.001).

<center>
![scoreage_avg](figures/worldnews_scoreage_avg.png)
</center>

There is more correlation with comment depth (r = -0.065) when using log10(score), but remains low:

<center>
![scoredepth_avg](figures/worldnews_scoredepth_avg.png)
</center>

Using box plots, we can see more clearly that the scores are sharply peaked at small values around 1.

<center>
![scoredepth_box](figures/worldnews_scoredepth_box.png)
</center>

Finally, some word clouds created from comments in /r/worldnews on a few different days of the month.

2015-05-04
![worldnews_wordcloud_05_04](figures/worldnews_wordcloud_05-04.png)

2015-05-13
![worldnews_wordcloud_05_13](figures/worldnews_wordcloud_05-13.png)

2015-05-18
![worldnews_wordcloud_05_18](figures/worldnews_wordcloud_05-18.png)

2015-05-29
![worldnews_wordcloud_05_29](figures/worldnews_wordcloud_05-29.png)
