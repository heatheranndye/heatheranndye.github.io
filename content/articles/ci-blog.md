title: Hello Data!
author: Heather Ann Dye
date: 10/26/2022
category: data science





## Hello data science! 

In this article, we'll create a small simulation and explore confidence intervals and the central limit theorem. 

We frequently talk about 95% or 90% confidence intervals but what information do these intervals give us?

#### The Central Limit Theorem

>Given a population with mean $\mu$ and standard deviation $\sigma$, the distribution of sample means $\left( \frac{\sum x}{n} = \bar{x} \right)$ with sample size $n$ will be approximately normally distributed for sufficiently large $n$. 

Given a few semesters of calculus and linear algebra, it can be shown that:

* $E(\bar{x}) = \mu$ 

* $Var(\bar{x})=\frac{\sigma^2}{n}$ 

Then, we hit another roadblock: *proving that for sufficiently large $n$, the sampling distribution is approximately normal.*  Even after going through the proof, it can be really difficult to think about what this means in practice. So, we'll construct some examples using Python.

We'll start by importing the packages used in this demonstration.


```python
import random 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from numpy import sqrt

```

#### Generate some data and build a confidence interval, Round 1

We'll start by generating some fake data from a continuous, uniform distribution on the interval $[0,10]$ with a sample size of 50. The mean, $\mu$, of this distribution is 5 and the standard deviation is approximately 2.9. The *sufficiently large* part of the Central Limit Theorem means that the sample size must be 30 or larger if the underlying distribution is not normal. I'll set the random seed so that this data is reproducible.


```python
random.seed(42)
mydata = pd.Series([random.uniform(0,10) for i in range(50)])
```

For right now, let's pretend that we don't know the specifics of the distribution that we sampled. (Because in reality, you wouldn't.)
First, inspect the five number summary using the describe method. Notice that the median ($\tilde{x}$) is 4.63 and the sample mean is 4.5. Based on the quartiles, this data set is symmetric. 

Next, we examine the sample standard deviation which is 2.9. This means that if 
the data comes from a normal distribution then about 68% of my data should fall in the interval $(1.6, 7.4)$. Using the describe function, I only have access to $P_{25}$ and $P_{50}$, so I'll investigate a little further. 


```python
mydata.describe()
```




    count    50.000000
    mean      4.506842
    std       2.932888
    min       0.064988
    25%       2.190886
    50%       4.636386
    75%       6.927794
    max       9.731158
    dtype: float64



We can check the quantiles of the data set to see that $P_{16} \approx 0.96$ and $P_{84} \approx 8.06$ This suggests that our data is not normal, but we can also visually confirm this by examining a histogram of the data. 


```python
mydata.quantile(q=0.16), mydata.quantile(q=0.84)
```




    (0.9608109148093343, 8.060286952634595)




```python
mydata.hist()
```




    <AxesSubplot: >




    
![png](ci-blog_files/ci-blog_9_1.png)
    


This visual inspection confirms that our data is not normal, but luckily! we are working with a sample of size 50. 

The formula for the confidence interval  of a sample mean is: $$\bar{x} \pm \frac{s}{\sqrt{n}} t_{\alpha /2, df}.$$  The $t_{\alpha /2, df}$ refers to a the critical value, a value on the x-axis of the Student-t distribution that cuts off a tail of area $\alpha /2$ and the $df$ refers to the degrees of freedom which specifies a specific t-distribution. 
*(Fun fact: The Student-t distribution was derived by William Seally Gossett who, at the time, was employed by the Guinness Brewery.)*

Python can quickly compute this formula. We use SciPy's t.ppf to obtain the critical value. We compute a 90% confidence interval, which means that
$$1-\alpha = 0.90$$ 
and 
$$\alpha /2  = 0.05.$$ 


```python
xbar = mydata.mean()
mysd = mydata.std()
(xbar - mysd/ sqrt(len(mydata)) * t.ppf(0.95, len(mydata)-1), xbar + mysd/ sqrt(len(mydata)) * t.ppf(0.95, len(mydata)-1))
```




    (3.8114545041480343, 5.202230430974262)



#### Confidence Intervals, Round 2

Now, we can investigate what exactly the confidence level means beyond selecting the critical value. The first thing that we'll do is write a small function to automatically compute our confidence interval.


```python
def student_t_confidence(data, conf_level):
    xbar = data.mean()
    mysd = data.std()/sqrt(len(data))
    alpha_2 = (1-conf_level)/2
    tsig = t.ppf(0.95, len(data)-1)
    return (xbar - mysd*tsig, xbar+mysd*tsig )
```

Here is our brand new function in action.


```python
student_t_confidence(mydata, 0.90)
```




    (3.8114545041480343, 5.202230430974262)



Next, we're going to generate 20 samples of size 50 and store them in a data frame.  

Here is a function that accomplishes this goal.


```python

def make_some_fake_data(sample_count: int, sample_size: int):
    local_df = pd.DataFrame()
    for i in range(sample_count):
        local_list =[random.uniform(0,10) for j in range(sample_size)]
        local_df[f"sample {i}"]=local_list
    return local_df
    

mydf = make_some_fake_data(20, 50)  

```

Here is the head of my new data frame.


```python
mydf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample 0</th>
      <th>sample 1</th>
      <th>sample 2</th>
      <th>sample 3</th>
      <th>sample 4</th>
      <th>sample 5</th>
      <th>sample 6</th>
      <th>sample 7</th>
      <th>sample 8</th>
      <th>sample 9</th>
      <th>sample 10</th>
      <th>sample 11</th>
      <th>sample 12</th>
      <th>sample 13</th>
      <th>sample 14</th>
      <th>sample 15</th>
      <th>sample 16</th>
      <th>sample 17</th>
      <th>sample 18</th>
      <th>sample 19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.701810</td>
      <td>0.114810</td>
      <td>9.951494</td>
      <td>9.689963</td>
      <td>3.161772</td>
      <td>9.039286</td>
      <td>9.811497</td>
      <td>4.421598</td>
      <td>8.972081</td>
      <td>6.222570</td>
      <td>6.906144</td>
      <td>0.856130</td>
      <td>1.843637</td>
      <td>6.476347</td>
      <td>3.232738</td>
      <td>2.828390</td>
      <td>8.723830</td>
      <td>9.926920</td>
      <td>7.242167</td>
      <td>0.993546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.095070</td>
      <td>7.207218</td>
      <td>6.498781</td>
      <td>9.263670</td>
      <td>7.518645</td>
      <td>5.455903</td>
      <td>5.362157</td>
      <td>2.137014</td>
      <td>7.436554</td>
      <td>0.269665</td>
      <td>6.272424</td>
      <td>7.200747</td>
      <td>0.513767</td>
      <td>9.084114</td>
      <td>9.701847</td>
      <td>2.985558</td>
      <td>0.358991</td>
      <td>2.952308</td>
      <td>9.799422</td>
      <td>6.856803</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.669778</td>
      <td>6.817104</td>
      <td>4.381001</td>
      <td>8.486957</td>
      <td>0.725431</td>
      <td>8.345950</td>
      <td>9.392371</td>
      <td>4.731862</td>
      <td>4.746744</td>
      <td>3.940203</td>
      <td>1.019013</td>
      <td>4.885778</td>
      <td>9.410636</td>
      <td>8.266312</td>
      <td>4.041751</td>
      <td>5.869377</td>
      <td>0.684207</td>
      <td>9.779445</td>
      <td>9.672697</td>
      <td>5.444659</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.366546</td>
      <td>5.369703</td>
      <td>5.175758</td>
      <td>1.663111</td>
      <td>4.582855</td>
      <td>5.825096</td>
      <td>1.153418</td>
      <td>9.011808</td>
      <td>2.591915</td>
      <td>5.643920</td>
      <td>7.724809</td>
      <td>7.581647</td>
      <td>4.777292</td>
      <td>0.714098</td>
      <td>5.145963</td>
      <td>9.989023</td>
      <td>6.311610</td>
      <td>6.582298</td>
      <td>8.045876</td>
      <td>9.778425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.480354</td>
      <td>2.668252</td>
      <td>1.210042</td>
      <td>4.856411</td>
      <td>9.984544</td>
      <td>1.480938</td>
      <td>9.704006</td>
      <td>7.960248</td>
      <td>2.472397</td>
      <td>0.271020</td>
      <td>8.502932</td>
      <td>6.906093</td>
      <td>8.221156</td>
      <td>1.659228</td>
      <td>9.881192</td>
      <td>4.896403</td>
      <td>9.209291</td>
      <td>2.744804</td>
      <td>3.657750</td>
      <td>3.586738</td>
    </tr>
  </tbody>
</table>
</div>



Here, we take our 20 samples and compute a confidence interval for each sample. The results  are stored in a dictionary, using the sample number 
as the key. 


```python
ci_dict = {}
for j in range(mydf.shape[1]):
    results = student_t_confidence(mydf.iloc[:,j], 0.9)
    ci_dict.update({j:results})

```

We analyze the results of our computations and count the number of intervals that actually contain the true population mean, 5. 
Our goal in computing these intervals is to estimate the true population mean.


```python
def count_the_good_ones(local_dict, truemean):
    count = 0 
    for i in range(len(local_dict)):
        therange = local_dict.get(i)
        if therange[0] < truemean and therange[1]> truemean:
            count +=1 
    return count 

thenum = count_the_good_ones(ci_dict, 5)    
print(thenum)
```

    17
    

And, *only 17 intervals* actually contain the true population mean. 

So what just happened?  Let's take a look at a visualization.


```python

plt.plot([5 for i in range(20)], color= 'k')
themins = [ci_dict.get(i)[0] for i in range(20)]
plt.plot(themins, 'bs')
themaxes = [ci_dict.get(i)[1] for i in range(20)]
plt.plot(themaxes, 'rs')
themeans = [(themins[i] + themaxes[i])/2 for i in range(20)]
plt.plot(themeans, 'D')
for i in range(20):
    plt.vlines(x=i, ymin=themins[i], ymax=themaxes[i])
```


    
![png](ci-blog_files/ci-blog_26_0.png)
    


The three intervals that do not contain the true population mean have sample means that are relatively far from the true sample mean. 
The sample means were in the tails of the sampling distribution, meaning that *in those cases*, we were unlucky and obtained a sample that did not represent the actual population that the sample was drawn from. 

This occurrs even if we collect all our data correctly.

With a 90% confidence interval, if you repeatedly collected samples of the same size and constructed confidence intervals, only 90% of the intervals contain the true population mean. 

In my simulation, we know the *true population mean* is five, so we can correctly pick out the *bad* confidence intervals. 

In reality, we have no way of knowing if our confidence interval is a *good* confidence interval or a *bad* confidence interval. 
