# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & NLP

### Description

In week four we've learned about a few different classifiers. In week five we'll learn about webscraping, APIs, and Natural Language Processing (NLP). This project will put those skills to the test.

For project 3, your goal is two-fold:
1. Using [Pushshift's](https://github.com/pushshift/api) API, you'll collect posts from two subreddits of your choosing.
2. You'll then use NLP to train a classifier on which subreddit a given post came from. This is a binary classification problem.


#### About the API

Pushshift's API is fairly straightforward. For example, if I want the posts from [`/r/boardgames`](https://www.reddit.com/r/boardgames), all I have to do is use the following url: https://api.pushshift.io/reddit/search/submission?subreddit=boardgames

To help you get started, we have a primer video on how to use the API: https://youtu.be/AcrjEWsMi_E

**NOTE:** Pushshift now limits you to 100 posts per request (no longer the 500 in the screencast).

---

### Executive Summary

- Late fall and early summer are an exciting time in the sports world. From August through September the NHL and NBA playoffs are in full swing, commanding the attention of American and global sports fans. 

- For the organizations with a vested interest in the games, whether they're a brand or the team itself, it's useful to understand public sentiment and who is saying what.

- To answer these questions we need to identify the tools and machine learning model that provide the most contextual understanding of the data.

---

### Task

- In the role of a researcher at a marketing firm that provides audience insights,  identify the ML model and NLP parameters that most accurately distinguish text blocks as being about Basketball or hockey.
- Use Reddit posts from the r/NHL and r/NBA subreddit threads as the baseline for training, testing and scoring your ML accuracy

---

### About the Data

- For this project we utilized [Reddit's Pushift API](https://github.com/pushshift/api), a RESTful API created by Reddit to help provide enhanced functionality and search capabilities for searching Reddit comments and submissions.

- The original dataset retrieved from the API includes the post, title and over 80 other columns of metadata or features related to the post. For the sake of initial learnings we'll be focused exclusively on the post title ('title') and text ('selftextl).

- To ensure enough data was available for training and testing after EDA our study began with requesting 3,000 posts from each of the NBA and NHL subreddits.

---

### Tools

- API: Reddit's Pushift API
- Importing/Cleaning:
    - pandas
    - numpy
    - matplotlib
- Vectorizers: **from** nltk.stem **import** WordNetLemmatizer, PorterStemmer
- Models:
    - LogisticRegression
    - KNeighborsClassifier
    - RandomForestClassifier

# The Process

#### Requests / EDA

To efficiently collect our data without overloading Reddit's server, our for loop pinged the NHL and NBA API endpoints, collecting 75 documents each time with a 3 second sleep timer, until reaching our 3,000 document target.

With the raw data collected  and exported to a new CSV, we utilized an entirely new notebook to:
- Concatenate NHL and NBA data into a single dataframe
- Remove unnecessary columns
- Drop nulls/duplicates
- Create new target column (NHL: 1, NBA: 0)
- Combine 'title' and 'selftext' data into a new 'combo' column
- Create 'combo_lem' column of lemmatized data (for later testing)
- Create 'combo_stem' column of stemmed data (for later testing)
  
The final 'nhl_nba_df.csv' file was then exported for our third and final notebook to evaluate model performance.

#### Modeling

The three models we're interested in testing are LogisticRegression, KNN and RandomForest. We'll first establish a baseline score for each without any advanced tuning.Once the top performer is identified we can use the param_grid and GridSearchCV() function to identify the best performing parameters within it.

However, before testing any models we'll add any/all words containing 'nhl', 'nba', 'hockey' or 'basketball' to our list of TF-IDF stop words. This will make the model more generalizable and reduce dependency on tell tale features.

After running all 3 models, RandomForestClassifier accounted for the highest degree of variance with a mean cross-val score of 0.87 (+/- 0.07).

| Model | Average Cross-Val Score | Confidence Interval |
| --- | --- | --- |
| **KNN** | 0.85 | +/-  0.10 |
| **Logistic Regression** | 0.82 | +/-  0.07 |
| **Random Forest** | 0.87 | +/-  0.07 |

#### Conclusions & Recommendations

Random Forest, an ensemble classifier, proved to be the strongest model and is the recommended approach for further NLP. By randomly selecting a subset of features at each split and identifying the best subset of features among the random splits, Random Forests can handle a large number or features, de-correlate the base trees and avoid overfitting. This makes it great for Natural Language Processing!

With that said, our Random Forrest score was only slightly better than KNN and Logistic Regression. Our next steps should include fine-tuning the hyperparameters in the GridSearchCV and exploring additional features from the original API request that could be of use. We should also spend time examining the words with high predictiveness as well as those that contributed to the 49 False-Negative predictions.  That insight will help inform next steps.

With our model identified we can now focus our efforts on fine-tuning it's accuracy and further examining the NBA and NHL posts themselves.

