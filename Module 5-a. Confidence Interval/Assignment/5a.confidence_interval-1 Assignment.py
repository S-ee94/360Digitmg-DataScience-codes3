# Basic Statistics (Module – 4 (Part – 1))
### Q1) Calculate probability from the given dataset for the below cases
        # Data_set: Cars.csv
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
df=pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-a. Confidence Interval\Assignment\cars.csv")
df.head()
### Calculate the probability of MPG of Cars for the below cases.
    
    # a.	P(MPG>38)
    # b.	P(MPG<40)
    # c.	P(20<MPG<50)
from scipy import stats
# from scipy.stats import norm
# a. P(MPG>38)
print("Probablity of MPG greater than 38 is {}".format(1-stats.norm.cdf(38,df.mpg.mean(),df.mpg.std())))
# b. P(MPG<40)
print("Probablity of MPG less than 40 is {}".format(stats.norm.cdf(40,df.mpg.mean(),df.mpg.std())))
# c. P(20<MPG<50)
print("Probablity of MPG between 20 and 50 is {}".format(stats.norm.cdf(50,df.mpg.mean(),df.mpg.std())-stats.norm.cdf(20,df.mpg.mean(),df.mpg.std())))


### Q2) Check whether the data follows normal distribution
    # a)	Check whether the MPG of Cars follows Normal Distribution Dataset:Cars.csv
    # b)	Check Whether the Adipose Tissue (AT) and Waist Circumference (Waist) from wc-at data set follows 
    # Normal Distribution
    # Dataset: wc-at.csv

### a)
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.distplot(df.mpg, label='mpg')
plt.xlabel('MPG')
plt.ylabel('Density')
plt.legend();
print("Mean is {}, Median is {}".format(df.mpg.mean(),df.mpg.median()))
# As mean and median have nearly equal values it is a normal distribution.

### b)
wat=pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-a. Confidence Interval\Assignment\wc-at.csv")
wat.head()
sns.distplot(wat.Waist, label='Waist')
plt.xlabel('Waist')
plt.ylabel('Density')
plt.legend();
print("Mean is {}, Median is {}".format(wat.Waist.mean(),wat.Waist.median()))
# As mean is nearly equal to median and also the data is fairly distributed we can say it is normally distributed.

sns.distplot(wat.AT, label='AT')
plt.xlabel('AT')
plt.ylabel('Density')
plt.legend();
print("Mean is {}, Median is {}".format(wat.AT.mean(),wat.AT.median()))
# As mean>median and data is positively skewed, it is not normal Distribution.


### Q3) Calculate the Z scores of 90% confidence interval,94% confidence interval, 60% confidence interval
z_score_90 = stats.norm.ppf(0.90, 0, 1) # Given probability, find the Z value
print("Z scores of 90% confidence interval : ", z_score_90)
z_score_94 = stats.norm.ppf(0.94, 0, 1) # Given probability, find the Z value
print("Z scores of 94% confidence interval : ", z_score_94)
z_score_60 = stats.norm.ppf(0.60, 0, 1) # Given probability, find the Z value
print("Z scores of 60% confidence interval : ", z_score_60)
### Q4) Calculate the t scores of 95% confidence interval, 96% confidence interval, 99% confidence interval for sample size of 25
t_score_95 = stats.t.ppf(0.95, 25) # Given probability, find the t value
print("t scores of 95% confidence interval : ", t_score_95)
t_score_96 = stats.t.ppf(0.96, 25) # Given probability, find the t value
print("t scores of 96% confidence interval : ", t_score_96)
t_score_99 = stats.t.ppf(0.99, 25) # Given probability, find the t value
print("t scores of 99% confidence interval : ", t_score_99)


### Q6) The time required for servicing transmissions is normally distributed with $\mu$= 45 minutes and $\sigma$= 8 minutes. The service manager plans to have work begin on the transmission of a customer’s car 10 minutes after the car is dropped off and the customer is told that the car will be ready within 1 hour from drop-off. What is the probability that the service manager cannot meet his commitment?
#Probability that car will be ready in less than one hour
mean_time = 45
std_time = 8
prob50 = stats.norm.cdf(50, mean_time, std_time)
print(r"P(Delivery in less than 50 minutes) : ", prob50 )
p_not = 1-prob50
print(r"P(Failure to Delivery in less than one hour) : ", p_not )
### Q10) Consider a company that has two different divisions. The annual profits from the two divisions are independent and have distributions Profit1 ~ N(5, 3^2) and Profit2 ~ N(7, 4^2) respectively. Both the profits are in \\$ Million. Answer the following questions about the total profit of the company in Rupees. Assume that \\$1 = Rs. 45
    # A.	Specify a Rupee range (centered on the mean) such that it contains95% probability for the annual profit of the company.
    # B.	Specify the 5th percentile of profit (in Rupees) for the company
    # C.	Which of the two divisions has a larger probability of making a loss in a given year?

# Mean profits from two different divisions of a company = Mean1 + Mean2
Mean = 5+7
print('Mean Profit is Rs', Mean*45,'Million')
import numpy as np
# Variance of profits from two different divisions of a company = SD^2 = SD1^2 + SD2^2
SD = np.sqrt((9)+(16))
print('Standard Deviation is Rs', SD*45, 'Million')

# A. Specify a Rupee range (centered on the mean) such that it contains 95% probability for the annual profit of the company.
print('Range is Rs',(stats.norm.interval(0.95,540,225)),'in Millions')

# B. Specify the 5th percentile of profit (in Rupees) for the company
# To compute 5th Percentile, we use the formula X=μ + Zσ; wherein from z table, 5 percentile = -1.645
X= 540+(-1.645)*(225)
print('5th percentile of profit (in Million Rupees) is',np.round(X,))

# C. Which of the two divisions has a larger probability of making a loss in a given year?
# Probability of Division 1 making a loss P(X<0)
stats.norm.cdf(0,5,3)
# Probability of Division 2 making a loss P(X<0)
stats.norm.cdf(0,7,4)
#### Inference: Probability of Division 1 making a loss in a given year is more than Division 2.
