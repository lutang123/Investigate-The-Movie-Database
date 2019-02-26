#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate The Movie Database (TMDb) 
# 
# --by Lu Tang
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > **Dataset**: This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue. Data can be download from[here](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dd1c4c_tmdb-movies/tmdb-movies.csv).
# > The final two columns ending with “_adj” show the budget and revenue of the associated movie in terms of 2010 dollars, accounting for inflation over time.
# 
# > **The project aims to explore the following questions:**
# > - Question 1: What are the trend for movie industry? Are movie industry making more money over years
# > - Question 2: Are newer movies more popular?
# > - Question 3: What kinds of properties are associated with movies that have high revenues?
# > - Question 4. Is it possible to make extremely high profit movies with low budget?
# > - Question 5: What are the top 10 rated movies? and how is their profitibility?

# In[36]:


# import library that will be used in this project 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[37]:


# loading data
tmdb=pd.read_csv('tmdb-movies.csv')
# show number of rows and columns
print(tmdb.shape)
# to avoid truncated output 
pd.options.display.max_columns = 150 
# show first 2 rows
tmdb.head(2)


# > **Initial observation**: 
# >- Our focus will be analyzing movie properties associated with high revenue, some columns are irrelavant for our analysis, e.g `id`,`imdb_id`, `homepage`, `tagline`, `keywords, overview, production_companies, release_date` (since we already have `release_year`).
# >- we can also remove `budget` and `revenue`, since we have `budget_adj` and `revenue_adj` to analyze.

# In[38]:


# Drop extraneous columns  
drop_col=['id','imdb_id','homepage','tagline','keywords','overview','production_companies','release_date','budget','revenue']
tmdb = tmdb.drop(drop_col, axis=1)
# check the result
tmdb.head(1)


# In[39]:


# check data type and missing values
tmdb.info()


# In[40]:


# check statistical information 
tmdb.describe(include='all')


# > **Insights**:
# >- Some columns contain NaN values, but the amount is not significant; we don't need to drop all the nulls at the beginning.
# >- Data type are all correct.
# >- The minimum `runtime` is 0, which is impossible, and some movies have extremely long runtime, we will investigate the outlier data
# >- `budget_adj` and `revenue_adj` have minimum and median value as 0 too, which is odd, and the difference from 75% to maximum is huge, we need to investigate in the later anaysis process
# >-  `popularity ` , `vote_count` has very uneven distribution, with some extreme high value data.  

# ### Data Cleaning 

# This dataseat is generally clean, column names are also clear and with preferred snakecase. For some string columns that contains '|', we will clean and analyze in the later part specific to the question we want to answer.

# **1. Remove duplicated data**

# In[41]:


# check how many rows are duplicated
sum(tmdb.duplicated())


# In[42]:


# Drop duplicated rows
tmdb.drop_duplicates(inplace=True)

# douch check the results
sum(tmdb.duplicated())


# **2. Cleaning abnormal data for runtime**

# In[43]:


# Find out how many rows are 0 for runtime
sum(tmdb["runtime"]==0)


# In[44]:


# Since it is impossible to have runtime as 0, we will remove these.
tmdb=tmdb[tmdb["runtime"]>0]

#double check the result
sum(tmdb["runtime"]==0)


# **3. Cleaning abnormal data for budget**

# In[45]:


sum(tmdb["budget_adj"]==0)


# In[46]:


# It is impossible to make a movie without any budget, we will remove these data
tmdb=tmdb[tmdb["budget_adj"]>0]

# Double check the result
sum(tmdb["budget_adj"]==0)


# **4. Cleaning abnormal data for revenue**

# In[47]:


sum(tmdb["revenue_adj"]==0)


# In[48]:


# It is impossible to make a movie without any budget, we will remove these data
tmdb=tmdb[tmdb["revenue_adj"]>0]

# Double check the result
sum(tmdb["revenue_adj"]==0)


# In[49]:


# Double check the cleaning result
print(tmdb.shape)
tmdb.head(1)


# <a id='eda'></a>
# ## Exploratory Data Analysis

# ## 1. Find pattern and visualize relationship

# **1_1. Explore relations with `revenue_adj`**

# In[50]:


# plot a heatmap to see correlation with `revenue_adj` for each columns
plt.figure(figsize=(8,5))
sns.heatmap(tmdb.corr(),annot=True,cmap='coolwarm')
plt.xticks(rotation=45)
plt.title('Correlation heatmap for whole movie data')


# **Conclusion:**
# >-  `revenue_adj` is positive-correlated with `popularity`, `vote_count` and `budget_adj`, which makes sense, the more popular, the more vote_count and and more revenues. And high budget movies are expected with high revenue too.
# >- `popularity` and `vote_count` are strongly correlated eith each other.
# >- `runtime`, `vote_average` and `release_year` do not have strong relation with any other columns. In fact `release_year` is slighly negative-correlated with `revenue_adj`. 

# **1_2. Plotting charts to find out the distribution for the variables that do not have strong correlation with Movie Revenue, i.e. `runtime, vote_average, release_year`.**

# In[51]:


# plotting distribution for 'runtime'
tmdb['runtime'].plot.hist(title='Runtime distribution for the whole movies data')
plt.xlabel('Movie Runtime')
plt.show()

# plotting distribution for 'vote_average'
tmdb['vote_average'].plot.hist(title='Vote_average distribution for the whole movies data')
plt.xlabel('Movie Vote_average')
plt.show()

# plotting distribution for 'vote_average'
tmdb['release_year'].plot.hist(title=('Movie Release_year Distribution'))
plt.show()


# **Conclusion:**
# >- Most movies have median length from about 100 minutes to 180 minutes.
# >- 'vote_average' has normal distribution, with average around 6.
# >- There are more movies produced over time.

# **1_3. Plotting scatter chart to explore detailed relationship between `popularity` and `vote_count`, and find out outliers.** 

# In[52]:


# plotting relation for 'popularity' and 'vote_count'
tmdb.plot.scatter(x='popularity',y='vote_count')


# **Conclusion:**
# >- From the scatter chart, we can confirm `popularity` and `vote_count` have strong positive correlation, same result as from the heatmap; however, we can also notice there are three movies rated extremely high popularity, but vote count is not extremely high.
# >- If we run regression model to decide movie revenues, we have to choose of one of them as an independent variable, but this is beyond the goal of this project.

# **1_4. Plotting scatter charts to furthur explore the relation with `revenue_adj` for the varibles of `popularity`, `vote_count` and `budget_adj`.**

# In[53]:


tmdb.plot.scatter(x='popularity', y='revenue_adj')
tmdb.plot.scatter(x='vote_count',y='revenue_adj')
tmdb.plot.scatter(x='budget_adj',y='revenue_adj')


# **Conclusion:**
# >- In general, the three variabels(`popularity`, `vote_count` and `budget_adj`) are all positively correlated with `revenue_adj`, but the correlation is not very strong, which is the same conclusion from the heatmap;
# >- There are many outlier data, some movies with extremely high popularity and high vote_count do not have extremely high revenue. These movies maybe controversial, and popularity and vote_count alone are not good indicator for movie success.
# >- Also, some extremely high budget movies do not have very high revenue, which means they maybe losing money.

# ## 2. Explore Answers for research questions 

# ### Question 1. What are the profitibility trend for movie industry? 

# In[54]:


2. # create a column for profit
tmdb['profit']=tmdb['revenue_adj']-tmdb['budget_adj']
tmdb['profit'].plot.box()


# >- **Some movies are losing money; others, however have huge profit.**

# In[55]:


# Create a dataframe that group by year and calculate mean value
year_mean=tmdb.groupby('release_year').mean()
year_mean.head(2)


# In[56]:


# plotting line chart for profit
plt.plot(year_mean.index, year_mean['profit'],label='Profit')
plt.xlabel('years')
plt.ylabel('In terms of 2010 dollars')
plt.title('Profit over years')
plt.legend()


# **Conclusion:**
# >- Overrall, the average profit per year is lower since 1980; There is less profit to making a movie compared with three decades ago.
# >- In the earlier years from 1960 to 1980, film industry have higher profit but with very high fluctuation too, and the profit trend is more stable in recent years.We can conclude in the earlier years, film industry is relatively new, high risk is associated with high profit.

# ### Question 2: Are newer movies more popular?

# In[57]:


# Since popularuty variable has some outlier data, we use vote_count to count for popularity.
plt.plot(year_mean.index, year_mean['vote_count'], label='vote_count')
plt.xlabel('years')
plt.ylabel('vote_count')
plt.title('Movie vote_count over years')
plt.legend()


# **Conclusion:**
# >- Yes. There is clear trend that the newer movies are more popular. 

# ### Question 3: What kinds of properties are associated with movies that have high revenues?

# **1. Find out the movies with very high revenues**

# In[58]:


# check the distribution for 'revenue_adj'
tmdb.revenue_adj.describe([.8,.9]).iloc[3:]


# In[59]:


# Create an ordinal data column to categorize movies with different levels of revenues
bin_edge=[0, 2.872138e+07, 1.496016e+08, 2.880722e+08, 3e+09]
bin_names=['very_low','low','high','very_high']
tmdb['revenue_level']=pd.cut(tmdb.revenue_adj,bin_edge,labels=bin_names)
tmdb['revenue_level'].value_counts()


# >- Half of the movies have very_low or even negative revenue
# >- There are **517 movies with very high_revenue**, let's focus on these movies.

# In[60]:


# Create a dataframe that only contains movies with very high revenues
very_high=tmdb[tmdb['revenue_level']=='very_high']
# View top 5 very_high revenue movies
very_high.head()


# **2. Find out among the very_high profit movies, what kinds of generes are the most common ones.** 

# In[61]:


# separate the generes with '|' and make it a new table
generes=very_high['genres'].str.split('|', expand=True)
# view the new table
generes.head()


# In[62]:


# Count the frequency for each genere for column 0, sorted as index.
col_0=generes.loc[:,0].value_counts().sort_index()
col_0


# In[63]:


# Convert the pandas Series to a data frame 
df_0=pd.DataFrame(data=col_0, index=col_0.index)
df_0


# In[64]:


# do the same for other columns:
col_1=generes.loc[:,1].value_counts().sort_index()
df_1=pd.DataFrame(data=col_1, index=col_1.index)

col_2=generes.loc[:,2].value_counts().sort_index()
df_2=pd.DataFrame(data=col_2, index=col_2.index)

col_3=generes.loc[:,3].value_counts().sort_index()
df_3=pd.DataFrame(data=col_3, index=col_3.index)

col_4=generes.loc[:,4].value_counts().sort_index()
df_4=pd.DataFrame(data=col_4, index=col_4.index)


# In[65]:


# join the other 4 dataframe together
generes_join=df_0.join(df_1).join(df_2).join(df_3).join(df_4)
generes_join.head()


# In[66]:


generes_join.fillna(0)


# In[67]:


# calculate the sum of each genere's frequency
generes_join['sum_generes']=generes_join.sum(axis=1)


# In[69]:


# sort value based on frequency sum and show top 5
generes_join.sort_values('sum_generes', ascending=False).head()


# >- Now we can see among these very_high revenue movies, the generes with **'Action', 'Adventure', 'Comedy','Drama', 'Thriller'** are top five generes

# **3. Explore movie runtime for very_high revenue movies**

# In[70]:


# check the runtime distribution
very_high['runtime'].plot.hist()
plt.xlabel('movie runtime')
plt.title('Runtime distribution for very high revenue movies')


# >- Most movies in this dataset have runtime ranging from **75 to 150 minutes, with 100 to 125 the most popular**. 
# >- Although some movies that have long runtime also have high revenue,a movie made under 1 hour is less likely to have very high revenue.

# **4. Explore release_year for very_high revenue movies**

# In[71]:


# check the release_year distribution
very_high['release_year'].plot.hist()
plt.xlabel('movie release_year')
plt.title('Release_year distribution for very high revenue movies')


# >- The movies in this dataset range from 1960 to 2015, which is very similiar distribution with the plot from whole dataset. 
# >- Since there are more movies produced over years and from the line chart previously, we can conclude newer movies does not mean higher profit, it will depend on the movie itself. 

# **5. Explore movie ratings for very_high revenue movies**

# In[72]:


# check the vote_average' distribution
very_high['vote_average'].plot.hist()
plt.xlabel('movie ratings')
plt.title('Movie ratings distribution for very high revenue movies')


# >- The chart reveals similar pattern compared with the plotting for whole dataset, but with higher rating in general ,ranging from 4.5 to 8.5, and average is roughly 6.5. From the heatmap earlier, we also see there isn't strong relation between vote_average and high revenue, but a movie with low rating will not have very high revenue.

# **6. Explore budget for very_high revenue movies**

# In[73]:


# Are these high revenue movies made with high budget?
very_high.plot.scatter(x='revenue_adj', y='budget_adj')
plt.title('Relation between revenue and budget')


# >- From the heatmap for movies with whole dataset, we found budget and revenue are positively related with each other, however, as we can see among these high_revenue movies, the correlation pattern is weak. 
# >- There are some high reveue moives with low budget and vice versa

# ### Question 4. Is it possible to make extremely high profit movies with low budget?

# In[74]:


# find out the median value of budget for the whole data
tmdb.budget_adj.median()


# In[75]:


# create a table for low_budget movies
low_budget=tmdb[tmdb['budget_adj']<30016111.9054567]
# check how many rows are in this table
print(low_budget.shape[0])
# view the distribution of 'revenue_level' in low_budget movie table
low_budget['revenue_level'].value_counts()


# In[76]:


# Percentage of very_high revenue movies with low_budget
61/1927


# >- There are only 0.03 of the movies in the low budget group but with very_high revenue, i.e most low budget movies does not yeild high revenue, but it does not mean it is imposible.

# In[89]:


# Create a table for these movies 
low_budget_very_high_revenue=low_budget[low_budget['revenue_level']=='very_high']
# check how many rows are in this table
print(low_budget_very_high_revenue.shape[0])
low_budget_very_high_revenue.head(3)


# In[90]:


# calculate the profit mean value in the category 
low_budget_very_high_revenue['profit'].mean()


# In[91]:


# We can also sort profit from the whole data to see the top 10 most profitable movies.
top_10_profit=tmdb.sort_values('profit',ascending=False).head(10)
top_10_profit


# In[92]:


# calculate the profit mean value in the categary 
top_10_profit['profit'].mean()


# In[93]:


# compare the difference between the average of top 10 movies's revenue and low_budget_very_high_revenue movie table
top_10_profit['profit'].mean()-low_budget_very_high_revenue['profit'].mean()


# **Conclusion:**
# >- Even though some low_budget movies made very_high revenues, the difference between the average revenue for top_10_profit movies and low_budget_very_high_revenue movies are huge.
# >- This can imply that for the movies made with huge profit, they are made with huge budget too. We can not expect a movie with low budget to make extremely high profit; however it is possible to make low budget movies with moderately high profit, but the chances are not significant, there are only 61 very_high revenue movies in the low_budget table.

# ### Question 5. What are the top 10 rated movies? and how is their profitibility?

# Since some movies have more vote_count, we can not directly compare a movie rated 10 with only 3 counts to the movie rated 7 with 100 counts. We will use IMDB'a definition to calculated weighted average for rating score.

# In[94]:


# m is the minimum votes required to be listed in the chart;
m= tmdb['vote_count'].quantile(0.9)
m


# In[95]:


# C is the mean vote across the whole report
C=tmdb['vote_count'].mean()
print(C)


# In[96]:


# Create a table for top 10% highest rated movies
q_movies = tmdb.copy().loc[tmdb['vote_count'] >= m]
q_movies.shape


# In[97]:


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# In[98]:


# show the top 10 rated movies
q_movies.sort_values('score',ascending=False).head(10)


# >- Above are the movies with highest rating score, as we can see, most have high or very_high revenue too, but it is different from the top_10_profit movies.

# In[99]:


q_movies['revenue_level'].value_counts()


# In[100]:


# Count for percentage of the one with low or very_low revenue
(64+5)/q_movies.shape[0]


# >- About 17% of the highly rated movies have low revenue, most have very_high revenue.
# >- This can confirm that a good rating score can yield high revenue.

# <a id='conclusions'></a>
# ## Conclusions
# 
# In this project we did comprehensive analysis on the movie database with a focus on movie revenues and other properties like generes and rating score. We did initial data exploration and answered all the questions 
# 
# **Summarize some of the featured findings:**
# >- In general, higher budget can yield higher revenues; movies made with low budget can have moderately high revenue, but successful rate is not very high.
# >- Popular movies have more vote_count and also have higher revenues, and newer movies are more popular
# >- Although much more movies are produced over time, movies industry is getting more stable and annual average profit is lower compared with movies made 3 decades ago.
# >-  'Action', 'Adventure', 'Comedy','Drama', 'Thriller' are the most common generes for movies with very_high revenue
# >-  Movies with high rating scores also have high revenues, but not nessesarily yield the highest or extremely high profit.
# 
# **Limitation of the project**
# >- The dataset we are using contains some incorrect information. As we see in data cleaning process, more than half of the data is deleted since it contains 0 value for runtime, budget or revenue. If we can have a more complete data, the analysis can be more accurate.
# >- Since the project is main focusing on movie revenue analysis, and we find popularity, budget and rating score can have impact on revenue, but these information is not enough to make revenue prediction as there might be other factors that can affect movie revenue but not included in this dataset.
