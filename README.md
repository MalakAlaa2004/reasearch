_# reasearch
science land AI bootcamp
Task 1 : research about statistical test:
*what is a statistical test ?
A statistical test provides a mechanism for making quantitative decisions about a process or processes.
*what is the intention of making a statistical test?
The intent is to determine whether there is enough evidence to "reject" a conjecture or hypothesis about the process.
*types of statistical test?
1-A|B test
2-Z test
3-t test
4-paired t test
5-chai square
____________________________________________________________________________________
A|B test 
*what is A|B testing ?
A/B testing is a user experience research methodology. A/B tests consist of a randomized experiment that usually involves two variants, although the concept can be also extended to multiple variants of the same variable
*example of A|B testing:
For example, you might send two versions of an email to your customer list (randomizing the list first, of course) and figure out which one generates more sales. Then you can just send out the winning version next time. Or you might test two versions of ad copy and see which one converts visitors more often 
Z Testing:
*What is a Z Test ?
A z-test is a statistical test to determine whether two population means are different when the variances are known and the sample size is large. A z-test is a hypothesis test in which the z-statistic follows a normal distribution. A z-statistic, or z-score, is a number representing the result from the z-test.
*types of z testing:
One-sample Z-test for means.
Two-sample Z-test for means.
One sample Z-test for proportion. 
Two sample Z-test for proportions.
  
T testing :
 ______________________________________________________________________
**difference between z test and t test:
  
paired t test:
*what is a paired t test?
The Paired Samples t Test compares the means of two measurements taken from the same individual, object, or related units. These "paired" measurements can represent things like: A measurement taken at two different times (e.g., pre-test and post-test score with an intervention administered between the two time points)
*example of paired t test:
For example, you would want to test the efficacy of a drug on the same group of patients before and after drug is given to the patients.
**difference between t test and paired t test:
sample t-test is used when the data of two samples are statistically independent, while the paired t-test is used when data is in the form of matched pairs.
chai square:
*What is a chai square test?
chi-squared test is a statistical hypothesis test used in the analysis of contingency tables when the sample sizes are large. In simpler terms, this test is primarily used to examine whether two categorical variables are independent in influencing the test statistic

 
 
*What is the difference between chi-square and Z test?
The Z-test is used when comparing the difference in population proportions between 2 groups. The Chi-square test is used when comparing the difference in population proportions between 2 or more groups or when comparing a group with a value.
Task 2 How to transform from any data type of distribuion to normal distribution?
*Should I transform data to normal distribution?
 
We know that in the regression analysis the response variable should be normally distributed to get better prediction results. Most of the data scientists claim they are getting more accurate results when they transform the independent variables too. It means skew correction for the independent variables
*1. Log Transformation :
import numpy as np
log_target = np.log1p(df["Target"])

*2. Square-Root Transformation 
sqrt_target = df["Target"]**(1/2)

*3. Reciprocal Transformation :
reciprocal_target = 1/df["Target"]

*4. Box-Cox Transformation:
 
from scipy.stats import boxcox
bcx_target, lam = boxcox(df["Target"])
#lam is the best lambda for the distrion.
*5. Yeo-Johnson Transformation:
from scipy.stats import yeojohnson
yf_target, lam = yeojohnson(df["TARGET"])


task 3:gradient descent local min 
“how to solve local minimum problem with gradient descent”
for each iteration keep on updating the value of x based on the gradient descent formula. From the above three iterations of gradient descent, we can notice that the value of x is decreasing iteration by iteration and will slowly converge to 0 (local minima) by running the gradient descent for more iterations.

Algorithm for Gradient Descent
Steps should be made in proportion to the negative of the function gradient (move away from the gradient) at the current point to find local minima. Gradient Ascent is the procedure for approaching a local maximum of a function by taking steps proportional to the positive of the gradient (moving towards the gradient).
repeat until convergence
{
    w = w - (learning_rate * (dJ/dw))
    b = b - (learning_rate * (dJ/db))
}
_____________________________________________________________________________task 4:stochastic gradient descent:
stochastic gradient descent is an optimization algorithm often used in machine learning applications to find the model parameters that correspond to the best fit between predicted and actual outputs. It's an inexact but powerful technique. Stochastic gradient descent is widely used in machine learning applications.



Gradient Descent — the algorithm
I use linear regression problem to explain gradient descent algorithm. The objective of regression, as we recall from this article, is to minimize the sum of squared residuals. We know that a function reaches its minimum value when the slope is equal to 0. By using this technique, we solved the linear regression problem and learnt the weight vector. The same problem can be solved by gradient descent technique.
“Gradient descent is an iterative algorithm, that starts from a random point on a function and travels down its slope in steps until it reaches the lowest point of that function.”

Stochastic Gradient Descent (SGD)
There are a few downsides of the gradient descent algorithm. We need to take a closer look at the amount of computation we make for each iteration of the algorithm.
Say we have 10,000 data points and 10 features. The sum of squared residuals consists of as many terms as there are data points, so 10000 terms in our case. We need to compute the derivative of this function with respect to each of the features, so in effect we will be doing 10000 * 10 = 100,000 computations per iteration. It is common to take 1000 iterations, in effect we have 100,000 * 1000 = 100000000 computations to complete the algorithm. That is pretty much an overhead and hence gradient descent is slow on huge data.
Stochastic gradient descent comes to our rescue !! “Stochastic”, in plain terms means “random”.
Where can we potentially induce randomness in our gradient descent algorithm??
Yes, you might have guessed it right !! It is while selecting data points at each step to calculate the derivatives. SGD randomly picks one data point from the whole data set at each iteration to reduce the computations enormously.
It is also common to sample a small number of data points instead of just one point at each step and that is called “mini-batch” gradient descent. Mini-batch tries to strike a balance between the goodness of gradient descent and speed of SGD.
____________________________________________________________________________
Task 5 conditional probability vs Naïve bayers probability
When to use each one of them?
*What is the difference between conditional probability and Bayes probability?
 
Conditional probability is the likelihood of an outcome occurring, based on a previous outcome having occurred in similar circumstances. Bayes' theorem provides a way to revise existing predictions or theories (update probabilities) given new or additional evidence.
*when to use ?
Bayes' theorem provides a way to revise existing predictions or theories given new or additional evidence. In finance, Bayes' theorem can be used to rate the risk of lending money to potential borrowers. In everyday situations, conditional probability is a probability where additional information is known. Finding the probability of a team scoring better in the next match as they have a former Olympian for a coach is a conditional probability compared to the probability when a random player is hired as a coach. 


