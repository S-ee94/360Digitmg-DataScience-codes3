# Confidence Interval–2

### Q3) Suppose we want to estimate the average weight of an adult male in Mexico.
# We draw a random sample of 2,000 men from a population of 3,000,000 men and weigh them.
# We find that the average person in our sample weighs 200 pounds, and 
# the standard deviation of the sample is 30 pounds. Calculate 94%, 98%, 96% confidence interval?

import scipy.stats as stats
# from scipy.stats import norm
weight_mean = 200
weight_Std = 30
conf_int_94 = stats.norm.interval(0.94, weight_mean,weight_Std/(2000**0.5)) # Finding confidence interval
print("94% confidence interval : ", conf_int_94)
conf_int_98 = stats.norm.interval(0.98, weight_mean,weight_Std/(2000**0.5)) # Finding confidence interval
print("98% confidence interval : ", conf_int_98)
conf_int_96 = stats.norm.interval(0.96, weight_mean,weight_Std/(2000**0.5)) # Finding confidence interval
print("96% confidence interval : ", conf_int_96)
