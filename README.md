# <p style="text-align: center;">Comparing Machine Learning Classifier models applied on Direct Marketing Business applications</p>

<p align="center">
<img src = images/classifier.png width = 50%/>
</p>

# Business Understanding
Direct marketing campaigns by businesses like banks and other financial institutions can be much more effective than mass campaigns. But businesses have very limited resources to perform such direct marketing campaign and using BI and Data Mining techniques to enhance the quality of such campains is very important. [This](CRISP-DM-BANK.pdf) was a study done to look at BI and DM tools/techniques to increase the effectiveness of direct marketing campaigns.

## Business goal
- To find a model that can explain success of a contact 
    - Success is when the client subscribes the deposit
- To find what  the main characteristics that affect success.
- Help in a better management of the available resource (Human Effort, phone calls, time)
- Selection of high quality and affordable set of potential buying customers.

### Once high quality potential customers are identified

- Offer using direct marketing campaigns, attractive long-term deposit applications
- Good interest rates


### Need for improved efficiency
- Lesser contacts
- No of success

# Data Source
The dataset comes from the UCI Machine Learning repository [here](https://archive.ics.uci.edu/dataset/222/bank+marketing). The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaings. 

# Project
This project attempts to use the data from the source above and applies various ML classifiers and compares and contrasts them.

Link to Jupyter notebook : https://github.com/1kit/

# Data Understanding
The following input variables are present in the data provided

```
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)
```

The output variable (target) is
```
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

```

## Analysis of the Data
The data has a number of categorical features
```
RangeIndex: 41188 entries, 0 to 41187
Data columns (total 21 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             41188 non-null  int64  
 1   job             41188 non-null  object 
 2   marital         41188 non-null  object 
 3   education       41188 non-null  object 
 4   default         41188 non-null  object 
 5   housing         41188 non-null  object 
 6   loan            41188 non-null  object 
 7   contact         41188 non-null  object 
 8   month           41188 non-null  object 
 9   day_of_week     41188 non-null  object 
 10  duration        41188 non-null  int64  
 11  campaign        41188 non-null  int64  
 12  pdays           41188 non-null  int64  
 13  previous        41188 non-null  int64  
 14  poutcome        41188 non-null  object 
 15  emp.var.rate    41188 non-null  float64
 16  cons.price.idx  41188 non-null  float64
 17  cons.conf.idx   41188 non-null  float64
 18  euribor3m       41188 non-null  float64
 19  nr.employed     41188 non-null  float64
 20  y               41188 non-null  object 
dtypes: float64(5), int64(5), object(11)
memory usage: 6.6+ MB
```

There were no missing data. A very few number of duplicate entries were dropped.

Here is the spread of the input data with respect to the output classes

```
NO  - 88.7%
YES - 11.2%
```
<p align="center">
<img src = images/bar.png width = 50%/>
</p>

__NOTE__ : There is a significant amout of mismatch with only 11% of the data resulted in the customer accepting the product.


# Data Preparation
Before we use the data, here are some encoding done on the input features

## Categorical Features
Here are some unique values for the categorical features

```
job         =>  ['housemaid' 'services' 'admin.' 'blue-collar' 'technician' 'retired'
                 'management' 'unemployed' 'self-employed' 'unknown' 'entrepreneur'
                 'student']
marital     =>  ['married' 'single' 'divorced' 'unknown']
education   =>  ['basic.4y' 'high.school' 'basic.6y' 'basic.9y' 'professional.course'
                 'unknown' 'university.degree' 'illiterate']
default     =>  ['no' 'unknown' 'yes']
housing     =>  ['no' 'yes' 'unknown']
loan        =>  ['no' 'yes' 'unknown']
contact     =>  ['telephone' 'cellular']
month       =>  ['may' 'jun' 'jul' 'aug' 'oct' 'nov' 'dec' 'mar' 'apr' 'sep']
day_of_week =>  ['mon' 'tue' 'wed' 'thu' 'fri']
poutcome    =>  ['nonexistent' 'failure' 'success']
y           =>  ['no' 'yes']
```

__Perform the following simple transformations__
- __Column `default`__ : Treat no and unknown as 0 and yes as 1
- __Column `housing`__ : Treat no and unknown as 0 and yes as 1
- __Column `loan`__ : Treat no and unknown as 0 and yes as 1
- __Drop column `contact`__ : It does not seem very interesting as a predictor
- __Drop column `poutcome`__ : This does not seem very interesting as a predictor
- __Column `y`__ : Treat no as 0 and yes as 1
- __Other Categorical Features__ : Do a 1-hot encoding for the rest of the categorical features

## Non-categorical Features
- __Column `pdays`__ : 96% of the pdays column is at 999. We can drop this
- __Column `duration`__ : As specified it highly affects the output target and cannot be known apriori.

# Modeling
The input data was split into a train/test for holdout testing. The test data represents __25%__ of the overall data. The `stratify` option was used to ensure that the baseline model accuracy is same for both test and training data

## Baseline model
A baseline model is one which predicts the majority class for all data points.
The accuracy score for the baseline (on both test/train data) = `88.7%`

## Simple model comparisons
Initially four simple classifier models were trained and here are the results

<p align="center">
<img src = images/simple_table.png width = 50%/>
</p>


## Improving the Models
To improve the models, we need to consider what is important for this use-case.

### Hyper-parameter tuning
As part of grid-search, we need to optimize on reducing both `FalsePositives (FP)` and `FalseNegatives (FN)`. This is because of the following:

- __Cost of FP__ : This could be the marketing spend wasted on targeting customers who dont take the product
- __Cost of FN__ : This could be the potential revenue lost from missing out on customers who would take the product.

The above requires that we tradeoff between `precision-recall`. The alternate way this can be achieved is by comparing area under the ROC curve. Hence the roc_auc option of scoring is used in GridSearch.

### Feature Engineering
- One way to do feature engineering is to use SelectFromModel() with LogisticRegression and L1 regularization to affect 
  weights for features. This can be incorporated into the pipeline
  
- The correlation matrix can also give us a good way to identify features that really matter to the output variable.

Here is a correlation heatmap
<p align="center">
<img src = images/heatmap.png width = 100%/>
</p>


From the correlation matrix, the following features are in decreasing order of correlation with `y`
```
nr.employed                      0.354669
euribor3m                        0.307740
emp.var.rate                     0.298289
previous                         0.230202
month_mar                        0.144027
month_oct                        0.137538
cons.price.idx                   0.136134
month_sep                        0.126079
month_may                        0.108278
job_student                      0.093962
job_retired                      0.092364
month_dec                        0.079311
job_blue-collar                  0.074431
campaign                         0.066361
cons.conf.idx                    0.054802
marital_single                   0.054209
education_university.degree      0.050267
education_basic.9y               0.045152
marital_married                  0.043476
month_jul                        0.032344
job_services                     0.032262
age                              0.030381
education_basic.6y               0.023493
education_unknown                0.021476
day_of_week_mon                  0.021241
job_entrepreneur                 0.016651
job_unemployed                   0.014749
day_of_week_thu                  0.013797
housing                          0.011804
month_nov                        0.011779
month_jun                        0.009193
month_aug                        0.008778
day_of_week_tue                  0.008123
education_high.school            0.007408
education_illiterate             0.007246
job_housemaid                    0.006510
day_of_week_wed                  0.006290
job_technician                   0.006069
marital_unknown                  0.005210
job_self-employed                0.004668
loan                             0.004478
default                          0.003042
education_professional.course    0.001071
job_management                   0.000426
job_unknown                      0.000154
```

__NOTE-1__: If performance is a concern, we could eliminate features that dont show too much correlation with `y`. I did not do that in this project.

__NOTE-2__: I also created a feature extractor that would run a `L1` regularization to select features. But I ended up not using it because it made the GridSearch very slow

```
extractor = SelectFromModel(LogisticRegression(penalty='l1', solver = 'liblinear' ,random_state = 42))
```

# Model Evaluation
A GridSearch was done over various parameters for the four models compared earlier. The GridSearch attempted to maximize on `roc_auc` (area under the ROC curve) as a way to tradeoff between recall and precision.


## Classification Reports and Confusion Matrices
### KNN
```
Classification Report: KNN
              precision    recall  f1-score   support

           0       0.90      0.99      0.94      9134
           1       0.59      0.10      0.17      1160

    accuracy                           0.89     10294
   macro avg       0.74      0.55      0.56     10294
weighted avg       0.86      0.89      0.86     10294
```
<p align="center">
<img src = images/knn_confusion.png width = 100%/>
</p>

### Decision Tree
```
Classification Report: DecisionTree
              precision    recall  f1-score   support

           0       0.91      0.98      0.94      9134
           1       0.56      0.24      0.34      1160

    accuracy                           0.89     10294
   macro avg       0.74      0.61      0.64     10294
weighted avg       0.87      0.89      0.87     10294
```
<p align="center">
<img src = images/dt_confusion.png width = 100%/>
</p>

### Logistic Regression
```
Classification Report: Logistic Regression
              precision    recall  f1-score   support

           0       0.90      0.98      0.94      9134
           1       0.53      0.14      0.22      1160

    accuracy                           0.89     10294
   macro avg       0.72      0.56      0.58     10294
weighted avg       0.86      0.89      0.86     10294
```
<p align="center">
<img src = images/log_confusion.png width = 100%/>
</p>

### SVM
```
Classification Report: SVM
              precision    recall  f1-score   support

           0       0.90      0.98      0.94      9134
           1       0.57      0.18      0.28      1160

    accuracy                           0.89     10294
   macro avg       0.74      0.58      0.61     10294
weighted avg       0.87      0.89      0.87     10294
```
<p align="center">
<img src = images/svm_confusion.png width = 100%/>
</p>


## ROC Curves
Here are the ROC curves for the models evaluated above

<p align="center">
<img src = images/roc.png width = 100%/>
</p>

## Model Comparison

<p align="center">
<img src = images/grid_table.png width = 100%/>
</p>

Looking at the precision/recall tradeoff (using the area under the ROC curve)

- Both `Decision Tree` and `Logistic Regression` perform quite well.
- The time taken to GridSearch and train for `Logistic Regression` was lower
- `KNN` did really great in terms of GridSearch+Train time and is not too bad in terms of `area under the curve`
- `SVM` did poorly in all respects

# Conclusion and Recommendations

Using one of the better performing models from above to predict new customers to target for direct marketing campaigns will lead to a much higher quality of customers. 
For future studies, the recommendation is to use models that improve the area under the ROC curve, which gives a good tradeoff.

It is believed that using these models would lead to:

- Effective campaigns
- Effective utilization of resources.
- Higher user engagement and conversion rates
- Improve business reputation and helps attract and retain customers
- Improve revenue 