# Introduction-to-Machine-learning-and-Data-analysis

Session 7&8
هدف:‌ آشنایی با روش های Resampling مانند CV و معرفی Regularization و رگرسیون های Lasso & Ridge

توجیه کردن روش های Resampling و بعد آماری آن
آشنایی با CV, Grid Search, Bootstrapping
حل مثال عملی برای یافتن Hyper parameter های مدل رگرسیونی
آشنایی با رگرسیون های Ridge & Lasso
حل مثال عملی با رگرسیون ها Ridge & Lasso


هدف: آشنایی با درخت های تصمیم گیری و جنگل تصادفی Random Forest & Decision Tree

آشنایی با تئوری های آماری مربوط به Decision Tree
حل مثال عملی با Decision Tree و بهینه سازی Hyper parameter های مدل
آشنایی با روش های Ensemble Learning و معرفی جنگل تصادفی
آشنایی با سایر الگوریتم‌های خردجمعی مانند Xgboost و Adaboost
مقایسه جنگل های تصادفی و درخت تصمیم روی دیتای واقعی
ریلیز تمرین سری سه


# Hyperparameter Optimization
hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm.  
* A **hyperparameter** is a parameter whose value is used to control the learning process.

### What is the difference between parameter and hyperparameter ❓ 

![image info](https://miro.medium.com/max/1400/1*9qdlqKa4dtjwXNDNaWe9UQ.jpeg)
<img src = "https://i.postimg.cc/5tCrG5Md/hyperparameters.png" />

* **Model parameters:** These are the parameters that are estimated by the model from the given data. For example the coefficients in a linear regression or logistic regression.
* **Model hyperparameters:** These are the parameters that cannot be estimated by the model from the given data. These parameters are used to estimate the model parameters. For example, the learning rate in deep neural networks.


### How To Fine-Tune Your Models

We have already explained about <span style="color:red">**Overfitting**</span>

One of the significant aspects of training your machine learning model is avoiding overfitting. This happens because your model is trying too hard to capture the noise in your training dataset. By noise, we mean the data points that don’t represent the actual properties of your data but random chance. Learning such data points makes your model more flexible, at the risk of `overfitting`.

* Adds a penalty term to the least squares loss function:

$$\large\mathcal{L}_{Ridge} = \sum_{n=1}^{N} (y_n-(\mathbf{w}\mathbf{x_n} + w_0))^2 + \alpha \sum_{i=1}^{p} w_i^2$$ 

* Model is penalized if it uses large coefficients ($w$)
    * Each feature should have as little effect on the outcome as possible 
    * We don't want to penalize $w_0$, so we leave it out
    * Called L2 regularization because it uses the L2 norm: $\sum w_i^2$
* The strength of the regularization can be controlled with the $\alpha$ hyperparameter.
    * Increasing $\alpha$ causes more regularization (or shrinkage). Default is 1.0.
### Regularization

Regularization is a way to avoid ``overfitting`` by penalizing high-valued regression coefficients. In simple terms, it reduces parameters and shrinks (`simplifies`) the model. In other words, this technique forces us not to learn a more complex or `flexible model`, to avoid the problem of overfitting.
This is a technique to minimize the complexity of the model (we will see what we mean by that) by penalizing the loss function to solve overfitting.


## L2 Ridge Regression

In a multiple `LinearRegression`, there are many variables at play. This sometimes poses a problem of choosing the wrong variables for the ML, which gives undesirable output as a result. Ridge regression is used in order to overcome this. This method is a `Regularization` technique in which an extra variable (tuning parameter) is added and optimised to offset the effect of multiple variables in LinearRegression (in the statistical context, it is referred to as `noise`).
The main idea of Ridge Regression is to fit a new line that `doesn’t` fit the training data. In other words, we introduce a certain Amount on Bias into the new trend line.

Please look at the following image:
<img src="https://i.postimg.cc/0ymd1Vrf/ridge.png" width='600'/>


## L1 Lasso Regression

### Lasso (Least Absolute Shrinkage and Selection Operator)

* Adds a different penalty term to the least squares sum:
$$\large\mathcal{L}_{Lasso} = \sum_{n=1}^{N} (y_n-(\mathbf{w}\mathbf{x_n} + w_0))^2 + \alpha \sum_{i=1}^{p} |w_i|$$ 
* Called L1 regularization because it uses the L1 norm
    * Will cause many weights to be exactly 0
* Same parameter $\alpha$ to control the strength of regularization. 

>**❗ NOTE:**  
>* L1 prefers coefficients to be exactly zero (sparse models)
>* Some features are ignored entirely: automatic feature selection
 
 
 >**NOTE:** Looking at the correlations between the features, we can see that there is some collinearity between the features. In particular, cylinders and displacement are highly correlated, and displacement and horsepower are highly correlated. 

LASSO is **more likely to remove features in sets of correlated features**, so it won’t be surprising if LASSO removes one or two of the features cylinders, displacement, weight and horsepower. On the other hand, LASSO selects the regularization strength, the strength of the penalty, based on cross-validated metrics. So, if there is some additional explanatory power behind a feature, it likely won’t be removed even if it is closely correlated to another feature.


### Hyperparameter tuning

There are several approaches to hyperparameter tuning, although two of the simplest and most common methods are random search and grid search.

* **Grid Search:**  set up a grid of hyperparameter values and for each combination, train a model and score on the validation data. In this approach, every single combination of hyperparameters values is tried which can be very inefficient!    
* **Random search:**  set up a grid of hyperparameter values and select random combinations to train the model and score. The number of search iterations is set based on time/resources.

<img src="https://i.postimg.cc/hP72yt5Z/gridvsrandom1.png" />
<img src="https://i.postimg.cc/5NcrMzLW/gridvsrandom2.png" />


* The curves on the left and on the top denote model accuracy


### Grid Search

- For each hyperparameter, create a list of interesting/possible values
    - E.g. For kNN: k in [1,3,5,7,9,11,33,55,77,99]
    - E.g. For SVM: C and gamma in [$10^{-10}$..$10^{10}$]
- Evaluate all possible combinations of hyperparameter values
    - E.g. using cross-validation
- Split the training data into a training and validation set
- Select the hyperparameter values yielding the best results on the validation set



### Random Search

- Grid Search has a few downsides:
    - Optimizing many hyperparameters creates a combinatorial explosion
    - You have to predefine a grid, hence you may jump over optimal values
- Random Search:
    - Picks `n_iter` random parameter values
    - Scales better, you control the number of iterations
    - Often works better in practice, too
        - not all hyperparameters interact strongly
        - you don't need to explore all combinations
        
# Resampling method

Resampling method is a tool consisting in repeatedly drawing samples from a dataset and calculating statistics and metrics on each of those samples in order to obtain further information about something, in the machine learning setting, this something is the performance of a model.  
  
Two commonly used resampling methods that you may encounter are k-fold cross-validation and the bootstrap.

* **Bootstrap.** Samples are drawn from the dataset with replacement (allowing the same sample to appear more than once in the sample), where those instances not drawn into the data sample may be used for the test set.
* **k-fold Cross-Validation.** A dataset is partitioned into k groups, where each group is given the opportunity of being used as a held out test set leaving the remaining groups as the training set.


## Cross-validation

You’re always ensuring your model performs well on your validation set, but what if your validation set is a little biased? Or worse, your training set is biased? To counter these possibilities, you can use cross validation.


## The Bootstrap
- Sample _n_ (dataset size) data points, with replacement, as training set (the bootstrap)
    - On average, bootstraps include 66% of all data points (some are duplicates)
- Use the unsampled (out-of-bootstrap) samples as the test set
- Repeat $k$ times to obtain $k$ scores


<img src="https://i.postimg.cc/PJcHYWG6/bootstrap.png" width="500"/>


The code results in creating an imbalanced dataset with 212 records labeled as malignant class reduced to 30. Thus, the total records count becomes benign tumour (357) + malignant tumour (30).

Next step is to use **resample** method to **oversample the minority class** (malignant tumour records in this example) and **undersample the majority class** (benign tumour records).

### Resample method for Over Sampling Minority Class
