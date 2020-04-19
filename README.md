
# Credit card fraud detection

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

The dataset that I'm going to use contains transactions made by credit cards in September 2013 by european cardholders.
See [Kaggle dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

Due to confidentiality issues, the original features and more background information about the data cannot be provided. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are `Time` and `Amount`. Feature `Time` contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature `Amount` is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature `Class` is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Imports

For the plots I'm going to use a custom made matplotlibrc style sheet, special thanks to [Jonny Brooks-Bartlett](https://towardsdatascience.com/a-new-plot-theme-for-matplotlib-gadfly-2cffc745ff84)


```python
import matplotlib.pyplot as plt
plt.style.use('gadfly')
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
```

## Exploratory Data Analysis (EDA)


```python
df = pd.read_csv('./dataset/creditcard.csv')
```


```python
df.head()
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



I want to inspect the summary statistics for each column available.
To achieve this, I define a function (see [link](https://medium.com/swlh/a-complete-guide-to-exploratory-data-analysis-and-data-cleaning-dd282925320f)), that takes my data as an input, and returns a data frame where each feature in my data set is now a row and the summary statistics are columns. The function will take a data frame as an input and calculate summary statistics to reveal insights about the data.


```python
def cols_eda(df): 
    eda_df = {}
    eda_df['null_sum'] = df.isnull().sum()
    eda_df['null_%'] = df.isnull().mean()
    eda_df['dtypes'] = df.dtypes
    eda_df['count'] = df.count()
    eda_df['mean'] = df.mean()
    eda_df['median'] = df.median()
    eda_df['min'] = df.min()
    eda_df['max'] = df.max()
    
    return pd.DataFrame(eda_df)
```


```python
cols_eda(df)
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
      <th>count</th>
      <th>dtypes</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>min</th>
      <th>null_%</th>
      <th>null_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Time</th>
      <td>284807</td>
      <td>float64</td>
      <td>172792.000000</td>
      <td>9.481386e+04</td>
      <td>84692.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V1</th>
      <td>284807</td>
      <td>float64</td>
      <td>2.454930</td>
      <td>3.919560e-15</td>
      <td>0.018109</td>
      <td>-56.407510</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V2</th>
      <td>284807</td>
      <td>float64</td>
      <td>22.057729</td>
      <td>5.688174e-16</td>
      <td>0.065486</td>
      <td>-72.715728</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V3</th>
      <td>284807</td>
      <td>float64</td>
      <td>9.382558</td>
      <td>-8.769071e-15</td>
      <td>0.179846</td>
      <td>-48.325589</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V4</th>
      <td>284807</td>
      <td>float64</td>
      <td>16.875344</td>
      <td>2.782312e-15</td>
      <td>-0.019847</td>
      <td>-5.683171</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V5</th>
      <td>284807</td>
      <td>float64</td>
      <td>34.801666</td>
      <td>-1.552563e-15</td>
      <td>-0.054336</td>
      <td>-113.743307</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V6</th>
      <td>284807</td>
      <td>float64</td>
      <td>73.301626</td>
      <td>2.010663e-15</td>
      <td>-0.274187</td>
      <td>-26.160506</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V7</th>
      <td>284807</td>
      <td>float64</td>
      <td>120.589494</td>
      <td>-1.694249e-15</td>
      <td>0.040103</td>
      <td>-43.557242</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V8</th>
      <td>284807</td>
      <td>float64</td>
      <td>20.007208</td>
      <td>-1.927028e-16</td>
      <td>0.022358</td>
      <td>-73.216718</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V9</th>
      <td>284807</td>
      <td>float64</td>
      <td>15.594995</td>
      <td>-3.137024e-15</td>
      <td>-0.051429</td>
      <td>-13.434066</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V10</th>
      <td>284807</td>
      <td>float64</td>
      <td>23.745136</td>
      <td>1.768627e-15</td>
      <td>-0.092917</td>
      <td>-24.588262</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V11</th>
      <td>284807</td>
      <td>float64</td>
      <td>12.018913</td>
      <td>9.170318e-16</td>
      <td>-0.032757</td>
      <td>-4.797473</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V12</th>
      <td>284807</td>
      <td>float64</td>
      <td>7.848392</td>
      <td>-1.810658e-15</td>
      <td>0.140033</td>
      <td>-18.683715</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V13</th>
      <td>284807</td>
      <td>float64</td>
      <td>7.126883</td>
      <td>1.693438e-15</td>
      <td>-0.013568</td>
      <td>-5.791881</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V14</th>
      <td>284807</td>
      <td>float64</td>
      <td>10.526766</td>
      <td>1.479045e-15</td>
      <td>0.050601</td>
      <td>-19.214325</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V15</th>
      <td>284807</td>
      <td>float64</td>
      <td>8.877742</td>
      <td>3.482336e-15</td>
      <td>0.048072</td>
      <td>-4.498945</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V16</th>
      <td>284807</td>
      <td>float64</td>
      <td>17.315112</td>
      <td>1.392007e-15</td>
      <td>0.066413</td>
      <td>-14.129855</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V17</th>
      <td>284807</td>
      <td>float64</td>
      <td>9.253526</td>
      <td>-7.528491e-16</td>
      <td>-0.065676</td>
      <td>-25.162799</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V18</th>
      <td>284807</td>
      <td>float64</td>
      <td>5.041069</td>
      <td>4.328772e-16</td>
      <td>-0.003636</td>
      <td>-9.498746</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V19</th>
      <td>284807</td>
      <td>float64</td>
      <td>5.591971</td>
      <td>9.049732e-16</td>
      <td>0.003735</td>
      <td>-7.213527</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V20</th>
      <td>284807</td>
      <td>float64</td>
      <td>39.420904</td>
      <td>5.085503e-16</td>
      <td>-0.062481</td>
      <td>-54.497720</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V21</th>
      <td>284807</td>
      <td>float64</td>
      <td>27.202839</td>
      <td>1.537294e-16</td>
      <td>-0.029450</td>
      <td>-34.830382</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V22</th>
      <td>284807</td>
      <td>float64</td>
      <td>10.503090</td>
      <td>7.959909e-16</td>
      <td>0.006782</td>
      <td>-10.933144</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V23</th>
      <td>284807</td>
      <td>float64</td>
      <td>22.528412</td>
      <td>5.367590e-16</td>
      <td>-0.011193</td>
      <td>-44.807735</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V24</th>
      <td>284807</td>
      <td>float64</td>
      <td>4.584549</td>
      <td>4.458112e-15</td>
      <td>0.040976</td>
      <td>-2.836627</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V25</th>
      <td>284807</td>
      <td>float64</td>
      <td>7.519589</td>
      <td>1.453003e-15</td>
      <td>0.016594</td>
      <td>-10.295397</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V26</th>
      <td>284807</td>
      <td>float64</td>
      <td>3.517346</td>
      <td>1.699104e-15</td>
      <td>-0.052139</td>
      <td>-2.604551</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V27</th>
      <td>284807</td>
      <td>float64</td>
      <td>31.612198</td>
      <td>-3.660161e-16</td>
      <td>0.001342</td>
      <td>-22.565679</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>V28</th>
      <td>284807</td>
      <td>float64</td>
      <td>33.847808</td>
      <td>-1.206049e-16</td>
      <td>0.011244</td>
      <td>-15.430084</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Amount</th>
      <td>284807</td>
      <td>float64</td>
      <td>25691.160000</td>
      <td>8.834962e+01</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>284807</td>
      <td>int64</td>
      <td>1.000000</td>
      <td>1.727486e-03</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



good news is that there are no missing values and all our variables are numerical.
I will now explore how much the dataset is unbalanced.


```python
sns.countplot('Class', data=df)

cls_counts = df.Class.value_counts()
print(f"""
Classes distribution \n{cls_counts} \nThe percentage of fraudulent \
transaction is {round(cls_counts[1]*100/cls_counts.sum(),2)}%.
      """)
```

    
    Classes distribution 
    0    284315
    1       492
    Name: Class, dtype: int64 
    The percentage of fraudulent transaction is 0.17%.
          
    


![png](output_10_1.png)


The dataset is highly imbalanced, most of the transactions are non-fraud.
Therefore, I'm going to build a balanced dataset by random under-sampling the majority class. This will inevitably yield a loss of information.


```python
rus = RandomUnderSampler(random_state=42)
X, y = rus.fit_sample(df.drop(columns='Class'), df.Class)
df_balanced  = pd.concat([X,y],axis=1)
sns.countplot('Class', data=df_balanced)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2e739baa630>




![png](output_12_1.png)


With the balanced dataset we can now investigate the correlations between the various features of the dataset.
In the follwing I'm using `heatmap` from the `seaborn` library, where red and blue denote positive and negative correlation, respectively.
We observe, for example, that while features `V16`, `V17` and `V18` are highly correlated with each other and the other features, features `V19`...`V28` are not.


```python
corr = df_balanced.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(250, 9, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-0.7, vmax=0.7, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2e73985e5c0>




![png](output_14_1.png)


We want to classify wether a certain transaction is fraudulent or not. Therefore, let's take a deeper look at the correlations between the dataset features and the target variable `Class`.


```python
correlations = df_balanced.corrwith(df_balanced['Class']).iloc[:-1].to_frame()
correlations['abs'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('abs', ascending=False)[0]
fig, ax = plt.subplots(figsize=(10,20))
sns.heatmap(sorted_correlations[:15].to_frame(), cmap='coolwarm', annot=True, vmin=-0.75, vmax=0.75, ax=ax);
```


![png](output_16_0.png)


The results show that features `V14`, `V4`, `V11` and `V12` are the ones with the highest correlation to the `Class` variable, our target.

Let's visualize how the `Class` distribution looks like for these features:


```python
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, col in enumerate(sorted_correlations.index[:9]):
    sns.distplot(df_balanced[df_balanced.Class==0][col], label='0', ax=axes[i//3][i%3])
    sns.distplot(df_balanced[df_balanced.Class==1][col], label='1', ax=axes[i//3][i%3])
plt.legend()
plt.tight_layout()
```


![png](output_19_0.png)


For `V14` we observe that more negative values generally indicate that the transaction is fraudulent, whereas for `V4` higher values seem to denote a fraudulent transaction.

# Model training and evaluation

I'm going to split the original dataset in `train` and `test` sets, because I want to keep the reality scenario where fraudulent transactions are very rare while testing my model. I will then randomly undersample the `train` dataset and train the model on it.


```python
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='Class'), df.Class, test_size=0.2, random_state=42)
X_train_us, y_train_us = rus.fit_sample(X_train, y_train)
```

In this exercise I don't aim at getting the best model with the best performance, I just want to get good enough results to have a strarting point for discussion.

I'm going to use a logistic regression modell off the shelf using `sklearn` library.


```python
logreg = LogisticRegression()
logreg.fit(X_train_us,y_train_us)
predictions = logreg.predict(X_test)
```

Given the very strong imbalance of the dataset, using accuracy as a metric is not a useful choice (actually it depends on your goal, discussed later). 
Instead, I'm going to use the confusion matrix that is a very effective, yet simple, classification metric.


```python
confmat = confusion_matrix(y_test, predictions)
plt.figure(figsize=(7,7))
ax = sns.heatmap(confmat, annot=True, annot_kws={'size':20}, fmt=".0f", 
            linewidths=.5, square = True, cmap = 'Blues', cbar_kws={"shrink": .5})
plt.ylabel('Actual label', size=20)
plt.xlabel('Predicted label', size=20)
plt.tick_params(axis='both', labelsize=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
```


![png](output_27_0.png)



```python
print(classification_report(y_test, predictions))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.95      0.97     56864
               1       0.03      0.92      0.06        98
    
        accuracy                           0.95     56962
       macro avg       0.52      0.93      0.52     56962
    weighted avg       1.00      0.95      0.97     56962
    
    

Let me give a short explanation of the metrics and results above.
- `accuracy`: total number of correct predictions divided by the total number of predictions,
- `precision`: precision of class tells us how **trustable** is the prediction of the model when it predicts a data point to belong to that class,
- `recall`: recall of a class tells us **how well** the model is able to detect that class,
- `f1-score`: combines precision and recall of a class in one metric.
As we can see the precision of class 1 is very low while recall is very high. Given the high imbalance in our dataset this is to be expected.

There are few points that one should really take into account when dealing with such problems:
- We need to understand our goal, we need to identify and correctly state the problem. In this exercise one should have answered the following questions:
    - What has a bigger impact on our cost function? Would we prefer to catch a fraud while rising false alarms or would we prefer to minimize false alarms (credit card blocked)?
- Resampling methods must be used thoughtfully, knowing that they modify the dataset and therefore the reality.
- Evaluation metrics should be chosen carefully, we need to choose our metrics to get the best overview on how our model performs with regards to our goals.

I recommend the following article [link](https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28). It gives a very exhaustive explanation on how to tackle imbalanced datasets with machine learning.
