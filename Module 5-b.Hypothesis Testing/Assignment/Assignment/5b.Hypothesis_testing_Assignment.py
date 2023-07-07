# Hypothesis Testing
# A systematic procedure is used by the researchers to predict whether the results obtained from a study supports a particular theory that is related to the population is known as hypothesis testing. It uses the sample data in order to evaluate the hypothesis of the population. It is the statistical inference method that is used to test the significance of the proposed hypothesized relation between the population statistics or the parameters and their corresponding estimators of the sample. This test consists of two hypotheses, the null hypothesis, and the alternative hypothesis.
import pandas as pd
import scipy 
from scipy import stats
##### 1.) A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured? Analyse the data and draw inferences at 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions.   File: Cutlets.csv
### 2 sample T Test
#### when we want to compare between two data we used 2 sample t test as per given steps following
##### here y = continuous (diameter of cutlet)
##### x = discrete ( unit a , unit b)
#2 sample T Test 

########File: Cutlets.csv##########
# Load the data
Cutlets = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-b.Hypothesis Testing\Assignment\Datasets_Hypothesis Testing\Cutlets.csv")
Cutlets.head()
Cutlets.columns = ("unit_A", "unit_B")

### Step frist to check the data is normal or not 
# Normality Test
stats.shapiro(Cutlets.unit_A) # Shapiro Test
#pvalue=0.3199819028377533 so p > 0.05 - p High -> Fails to reject Ho
stats.shapiro(Cutlets.unit_B) # Shapiro Test
#pvalue=0.5224985480308533 so p > 0.05 - p High -> Fails to reject Ho

### External condition are same or not
# Dimensions of cutlets measured from two different Unit so External conditions are different So check with the variance test that Var(unit A) equal or not with Var( unit B)

### Step two to check the variance between the two different units
# Variance test
# scipy.stats.levene(Cutlets.unit_A,Cutlets.unit_B)
#pvalue=0.4176162212502553 so p > 0.05 - p High -> Fails to reject Ho

### Step three for 2 sample t test 
# 2 Sample T test
# scipy.stats.ttest_ind(Cutlets.unit_A,Cutlets.unit_B)
#pvalue=0.47223947245995 so p > 0.05 - p High -> Fails to reject Ho


##### 2.) A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch. Analyze the data and determine whether there is any difference in average TAT among the different laboratories at 5% significance level. File: LabTAT.csv
### One-way Anova test
##### this test is used when there Y is continuous and x is descrete
##### here y = Turn around time
##### x = 4 laboratories(lab1,lab2,lab3,lab4)

#importing dataset
Labtat = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-b.Hypothesis Testing\Assignment\Datasets_Hypothesis Testing\lab_tat_updated.csv")
Labtat.head()
Labtat.columns = ("lab_1", "lab_2","lab_3","lab_4")

### Step frist to check the data is normal or not 
#Normality Test
stats.shapiro(Labtat.lab_1)
#p-value = 0.4232 means p > 0.05 so p High-> Fail to reject Ho
stats.shapiro(Labtat.lab_2)
#p-value = 0.8637 means p > 0.05 so p High-> Fail to reject Ho
stats.shapiro(Labtat.lab_3)
#p-value = 0.06547 means p > 0.05 so p High-> Fail to reject Ho
stats.shapiro(Labtat.lab_4)
#p-value = 0.6619 means p > 0.05 so p High-> Fail to reject Ho

### Step two to check the varience between laboratories
#Variance Test
scipy.stats.levene(Labtat.lab_1, Labtat.lab_2,Labtat.lab_3, Labtat.lab_4)
#pvalue=0.38107 means p > 0.05 so p High-> Fail to reject Ho
### Step three for one-way anova test

#One way Anova test
F, P = stats.f_oneway(Labtat.lab_1, Labtat.lab_2, Labtat.lab_3, Labtat.lab_4)
P
# p-value = 2.143740909435053e-58 < 0.05 reject null hypothesis

# TAT reports of laboratories are different


##### 3) Sales of products in four different regions is tabulated for males and females. Find if male-female buyer rations are similar across regions. East West North South
##### Males 	50 	142 	131 	70 
##### Females 	550 	351 	480 	350
### Chi-squared test
##### This test used when there y is discrete and x is discrete
##### here y = (male and female)
##### x = (East, West, North, South) - Regoin
Buyer = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-b.Hypothesis Testing\Assignment\Datasets_Hypothesis Testing\BuyerRatio.csv")
Buyer.head()

#Rename of Column name
Buyer.rename(columns = {"Observed Values":"Gender"}, inplace = True)
Buyer.columns

#replacing the gender with 0 and 1
Buyer["Gender"].replace("Males", 0 , inplace = True)
Buyer["Gender"].replace("Females", 1 , inplace = True)
Buyer

Chisquare_results = scipy.stats.chi2_contingency(Buyer)
Chisquare_results 

Chi_square = [["Test statistics", "p-value"],[Chisquare_results[0],Chisquare_results[1]]]
Chi_square
#AS p = 0.7919942975413565 p > 0.05 so Fails to reject null hypothesis



##### 4.) Telecall uses 4 centers around the globe to process customer order forms. They audit a certain % of the customer order forms. Any error in order form renders it defective and must be reworked before processing. The manager wants to check whether the defective % varies by center. Please analyze the data at 5% significance level and help the manager draw appropriate inferences  File: Customer OrderForm.csv

### chi-squared test
##### y= {Error free , defective}
##### x= { Phillipines, Indonesia, Malta, India }
COF = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-b.Hypothesis Testing\Assignment\Datasets_Hypothesis Testing\CustomerOrderform.csv")
COF.head()

COF = pd.DataFrame(COF.unstack(level = -1)).reset_index().drop("level_1",axis = 1)
COF.head()

COF = COF.rename(columns = {"level_0":"Country" , 0: "Defective"})
COF.head()

count = pd.crosstab(COF["Country"], COF["Defective"])
count

Chisquares_results = scipy.stats.chi2_contingency(count)
Chisquares_results

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
#p=0.2771020991233144 -> p > 0.05 -> Fails to reject null hypothesis



##### 5.) Fantaloons Sales managers commented that % of males versus females walking into the store differ based on day of the week. Analyze the data and determine whether there is evidence at 5 % significance level to support this hypothesis.  File: Fantaloons.csv

### 2 proportion test
##### y = {Weekend , weekday}
##### x = {Female, Male}
import numpy as np
Fantaloons = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-b.Hypothesis Testing\Assignment\Datasets_Hypothesis Testing\Fantaloons.csv")
Fantaloons.head()

#Dropping null values
Fantaloons.dropna(inplace = True)
New_sale = pd.DataFrame(Fantaloons.unstack(level = -1)).reset_index().drop("level_1",axis = 1)
New_sale.head()

#reset_index() is used for all value in way means first take all weekdays and then weekend
New_sale = New_sale.rename(columns = {"level_0":"Day" , 0: "Gender"})
New_sale.head()

#using proportion test
from statsmodels.stats.proportion import proportions_ztest
tab1 = New_sale.Day.value_counts()
tab1
tab2 = New_sale.Gender.value_counts()
tab2

# crosstable table
pd.crosstab(New_sale.Day, New_sale.Gender)
count = np.array([233, 167])
nobs = np.array([520,280])
stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval) 
#p=6.261142877946052e-05 -> p < 0.05 -> reject null hypothesis
stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print(pval)  
#p=0.9999686942856103 -> p > 0.05 -> Fails to reject null hypothesis
