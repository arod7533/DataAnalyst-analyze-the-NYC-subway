
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import pandasql
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ggplot import *
import scipy
import scipy.stats
import math


# In[3]:

# This code is to read the csv file using Pandas

weather_data = pd.read_csv('/Users/TonyRodriguez1/Dropbox/Udacity/DataAnalyst/P2 Analyzing the NYC Subway Dataset/improved-dataset/turnstile_weather_v2.csv')
weather_data


# In[24]:

# I used this code to get a list of all columns in my dataframe

weather_data.columns.values


# In[10]:

# P2: Question 1
# This SQL query displays columns 'rain' and a count of column the amount of columns where 'rain' is equal to 1. 

q = """
    SELECT rain, COUNT(rain)
    FROM weather_data
    WHERE rain = 1;
    """
rainy_days = pandasql.sqldf(q.lower(), locals())
rainy_days


# In[11]:

# P2: Question 2
# This SQL query displays column 'fog' and the max temperature if there was not fog (0) and if there was fog (1)

q1 = """
    SELECT fog, MAX(tempi)
    FROM weather_data
    GROUP BY fog;
    """
foggy_days = pandasql.sqldf(q1.lower(), locals())
foggy_days


# In[12]:

# P2: Question 3
# This SQL query displays the average temperature on either Saturday (5) or Sunday (6); the average temperature 
# on weekends. 

q3 = """
    SELECT avg(meantempi)
    FROM weather_data
    WHERE day_week = 5
    OR day_week = 6;
    """
mean_temp_weekends = pandasql.sqldf(q3.lower(), locals())
mean_temp_weekends


# In[13]:

# P2: Question 4
# This SQL query displays the average temperature, grouped by days of the week, where temperature is greater than 55
# and it there was rain

q = """
    SELECT day_week,rain, avg(tempi)
    FROM weather_data
    WHERE rain = 1
    AND tempi > 55
    GROUP BY day_week;
    """
avg_min_temp_rainy = pandasql.sqldf(q.lower(), locals())
avg_min_temp_rainy


# In[14]:

# P2: Question 5 Part 1
# This python code opens turnstile_110507.txt as a csv and prints out the first few rows to see what information 
# I'm working with. I used 'with open' so that the file automatically closes after I finish. 

with open('/Users/TonyRodriguez1/Dropbox/Udacity/DataAnalyst/P2 Analyzing the NYC Subway Dataset/turnstile_110507.txt') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print (row[0], row[1], row[2])


# In[17]:

# P2: Question 5 Part 2
# This code designates the read file and the write file. The for loop iterates through each row in 'read_file'
# and prepends 3 constand columns, which is the specific turnstile information for all the data. It then adds
# columns for the range (start_index, end_index) and moves 5 columns for every iteration, thereby picking up all
# the data in the text file. It writes it to a new updated turnstile txt file. 

file_in = open('/Users/TonyRodriguez1/Dropbox/Udacity/DataAnalyst/P2 Analyzing the NYC Subway Dataset/turnstile_110507.txt')
file_out = open('/Users/TonyRodriguez1/Dropbox/Udacity/DataAnalyst/P2 Analyzing the NYC Subway Dataset/updated_' + 'turnstile_110507.txt', 'wb')
read_file = csv.reader(file_in, delimiter =',')
writer_out = csv.writer(file_out, delimiter =',', lineterminator='\n')

for row in read_file:
    const_col1 = row[0]
    const_col2 = row[1]
    const_col3 = row[2]
    list_size = len(row)
    start_index=3
    end_index=list_size
    for records in range(start_index, end_index, 5):
        out_line = [const_col1, const_col2, const_col3, row[records], row[records+1], row[records+2], row[records+3], row[records+4] ]
        writer_out.writerow(out_line)


# In[18]:

# P2: Question 6 Part 1
# This python code gives column names to column of data

filenames = ['/Users/TonyRodriguez1/Dropbox/Udacity/DataAnalyst/P2 Analyzing the NYC Subway Dataset/updated_turnstile_110507.txt']

with open('/Users/TonyRodriguez1/Dropbox/Udacity/DataAnalyst/P2 Analyzing the NYC Subway Dataset/master_data.txt', 'w') as outfile:
    outfile.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())


# In[19]:

# P2: Question 6 Part 2
# This code looks at my data to verify it came out how I wanted

turnstile_data = pd.read_csv('/Users/TonyRodriguez1/Dropbox/Udacity/DataAnalyst/P2 Analyzing the NYC Subway Dataset/master_data.txt')
turnstile_data


# In[20]:

# P2: Question 7
# This code filters out any data where DESCn does not equal 'REGULAR'

turnstile_data = turnstile_data[turnstile_data.DESCn=='REGULAR']
turnstile_data


# In[21]:

# P2: Question 8 & 9
# This code adds two columns which show the difference between ENTRIESn of the current row and of the previous row
# It also replaces any 'NaN' values with a 1 or a 0

turnstile_data['ENTRIESn_hourly'] = turnstile_data['ENTRIESn'] - turnstile_data['ENTRIESn'].shift(1)
turnstile_data.fillna(1, inplace=True)
turnstile_data['EXITSn_hourly'] = turnstile_data['EXITSn'] - turnstile_data['EXITSn'].shift(1)
turnstile_data.fillna(0, inplace=True)
turnstile_data


# In[22]:

# P2: Question 10
# This function returns the hour as an integer

def get_hour(row):    
    return int(turnstile_data['TIMEn'][row][1])
get_hour(1) 


# In[23]:

# P2: Question 11
# This function returns the data in the proper format

def get_format_time(row):
    date = turnstile_data['DATEn'][row]
    formatted_date = datetime.datetime.strptime(date, '%m-%d-%y')
    new_date = formatted_date.strftime('%Y-%m-%d')
    return new_date

get_format_time(1)


# In[65]:

# P3: Question 1
# Using matplotlib I created two histograms showing the amount of hourly entries into the subway when it was raining
# and when it was not raining.

plt.figure()
weather_data['ENTRIESn_hourly'][weather_data['rain']==0].plot(kind='hist', bins=20, range=(0,6000), alpha=0.5, label='No Rain')
weather_data['ENTRIESn_hourly'][weather_data['rain']==1].plot(kind='hist', bins=20, range=(0,6000), alpha=0.5, label='Rain')
plt.xlabel('ENTRIESn_hourly')
plt.ylabel('Frequency')
plt.title('NYC Subway Ridership Rain vs. No Rain')
plt.legend()
plt.show()


# In[34]:

# P3: Question 2
# The above distributions for Rain and No Rain are not normally distributed, so we cannot use Welch's t-Test because
# it assumes both samples will be approximately normally distributed. 


# In[8]:

# P3: Questions 3 & 4
# Mann-Whitney U test, aka Wilcoxon rank sum test, is a non-parametric test that can be used when the distribution of
# the data appears to be non-normal. Non-parametric tests make no assumption of the probability distribution of the data. 
#
# H0: P ( x > y) = 0.5
# Ha: P ( x > y) != 0.5
#
# Alpha= 0.05
#
# With a p-value of 5.4821391424874991e-06, I reject the null at the 5% significance level and can 
# assume that the distributions of the populations are not equal.

rain = weather_data['ENTRIESn_hourly'][weather_data['rain']==1]
no_rain = weather_data['ENTRIESn_hourly'][weather_data['rain']==0]
rain_mean = np.mean(rain)
no_rain_mean = np.mean(no_rain)
U, p = scipy.stats.mannwhitneyu(rain, no_rain)
rain_mean, no_rain_mean, U, p*2


# In[66]:

# P3: Questions 5, 6, 7, 8 
# Using OLS regression, I regressed 'UNIT' as dummy variables, 'hour', 'meantempi', 'weekday', 'precipi'
# predictor variables against my dependent variable (y) 'ENTRIESn_hourly'. The R^2 value of 0.483 means that 
# 48.3% of the variability in ENTRIESn_hourly can be explained by the variables in my regression model. 

def linear_regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    print results.summary()
    intercept = results.params[0]
    params = results.params[1:]    
    return intercept, params


def predictions(dataframe):
    features = weather_data[[ 'hour', 'meantempi', 'weekday', 'precipi', 'rain']]  
    dummy_units = pd.get_dummies(weather_data['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    values = weather_data['ENTRIESn_hourly']
    intercept, params = linear_regression(features, values)
    predictions = intercept + np.dot(features, params)
    return(predictions)
    
print predictions(weather_data)


# In[15]:

features = weather_data[['precipi', 'hour', 'meantempi','weekday']]
dummy_units = pd.get_dummies(weather_data['UNIT'], prefix='unit')
features = features.join(dummy_units)
features = sm.add_constant(features)
values = weather_data['ENTRIESn_hourly']
model = sm.OLS(values, features)
results = model.fit()

plt.figure()
data = results.resid
data.plot(kind='hist', bins=120)
plt.ylim((0,4500))
plt.xlim((-20000,20000))
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Residual plot')
plt.show()
data.describe()


# In[42]:

# P4: Question 1 Part 1
# Before I plotted by data using ggplot, I queried the dataset to extract specific data to plot
# This subset of data shows the sum of entries into the subway by day

q = """
SELECT day_week, sum(ENTRIESn_hourly)
FROM weather_data
GROUP BY day_week;
"""
results_q = pandasql.sqldf(q.lower(), locals())
results_q


# In[44]:

# P4: Question 1 Part 2
# I plotted a histogram using ggplot to show the amount of entries by day of the week. 0 being Monday, 6 being Sunday.
# There is clearly an increase in subway ridership on weekdays, with a sharp drop off on weekends.

print ggplot(results_q,aes('day_week', 'sum(entriesn_hourly)'))+geom_bar(stat='bar', fill="green")+scale_x_continuous(breaks=(0,1,2,3,4,5,6), limits=(-1,7))+xlab('Day of the week')+ylab('Entries into the subway')+ylim(0,16000000)+ggtitle('Entries into Subway by day of week')


# In[45]:

# P4: Question 1 Part 3
# This query pulls the date and sums up the hourly entries by date. 

q1 = """
SELECT DATEn, sum(ENTRIESn_hourly)
FROM weather_data
GROUP BY DATEn;
"""
results_q1 = pandasql.sqldf(q1.lower(), locals())
results_q1


# In[46]:

# P4: Question 1 Part 2
# I used ggplot to plot the time series of Subway Ridership during May of 2011. You can clearly see subway ridership 
# increases on weekdays and drops off significantly on weekends, which is consistent with the histogram shown above.
# May 30, 2011 was of interest to me because I could not figure out why subway ridership dropped off on a Monday. 
# After doing a quick google search, I found out that May 30, 2011 was Memorial Day, and it stands to reason that 
# subway ridership would be down. I did have trouble truncating the x axis ticks using ggplot. 

pd.options.mode.chained_assignment = None
print ggplot(results_q1, aes(pd.to_datetime(results_q1['DATEn']), 'sum(entriesn_hourly)'))+geom_point(color='green')+geom_line(color='green')+xlab('Date')+ylab('Entries into subway')+ggtitle('Subway Ridership')


# In[62]:

# P4: Question 2 Part 1
# This query pulls the weather conditions, sum of hourly entries, and temperature for my next plot. 

q2 = """
SELECT conds, sum(ENTRIESn_hourly), round(avg(tempi),0)
FROM weather_data
GROUP BY conds;
"""
results_q2 = pandasql.sqldf(q2.lower(), locals())
results_q2


# In[63]:

# P4: Question 2 Part 2
# This plot shows hourly entries into the subway on the y axis, weather conditions on the x axis, and uses colour 
# scaling to show the average temperature in the fill of the bars. As expected, days with rain and overcast were the
# coldest, and days where the weather was clear, it was hotter. I could not figure out how to change the legend 
# title. I think that using scale_color_manual may do the trick, but the documentation on the yhat website was not 
# working. Other than that, I was impressed  with how much functionality ggplot gives you to customize the 
# visualization.

print ggplot(results_q2,aes('conds', 'sum(entriesn_hourly)',fill='round(avg(tempi),0)'))+geom_bar(stat='bar')+xlab('Conditions')+ylab('Entries into the subway')+ylim(0,30000000)+ggtitle('Weather conditions with temperature')+theme(axis_text_x = element_text(angle = 45, hjust = 1))+scale_colour_gradient2(low='white', high='darkgreen')


# In[ ]:



