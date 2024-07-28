<img src="./keagdi05.png"
style="width:2.50417in;height:2.50417in" />

> **Wearable** **Stress** **and** **Affect** **Detection** **(WESAD)**
>
> **Final** **Project** **Proposal**
>
> **Group** **–** **24**
>
> Venkata Sai Prasad Aka Aditya Kondepudi
>
>
> [<u>aka.v@northeastern.edu</u>](mailto:aka.v@northeastern.edu)
> [<u>kondepudi.a@northeastern.edu</u>](mailto:kondepudi.a@northeastern.edu)

Percentage of Effort Contributed by Venkata: 50% Percentage of Effort
Contributed by Aditya: 50%

Submission Date: 12/09/2022

**<u>PROBLEM SETTING:</u>**

The major point that we are addressing in this project is to help users
improve their state of health by intimating medical practitioner their
state of stress or increased stress. An analysis of federal health data
found that 8.3 million American adults, or roughly 3.4 percent of the
population, experience severe psychological discomfort. The researchers
noted that 3 percent or less of Americans were thought to be
experiencing substantial psychological discomfort in previous
estimations. Hence, in this project we are trying to reduce their mental
stress and improving their Mental Health.

**<u>PROBLEM DEFINITION:</u>**

In this project, we use WESAD which is publicly available dataset for
Wearable Stress and Affect

Detection. This multimodal dataset features physiological and motion
data, recorded from both

a wrist- and a chest-worn device, of 15 subjects during a lab study. The
following sensor

modalities are included: blood volume pulse, electrocardiogram,
electrodermal activity,

electromyogram, respiration, body temperature, and three-axis
acceleration. The goal of the

project is to analyze the data of the population, and to predict whether
the person is feeling

stressed or not within a measuring range of five different levels.

**<u>DATA SOURCE CITATION:</u>**

This dataset was procured from UCI Machine Learning Repository

Data Source Citation:

[<u>https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29</u>](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)

**<u>DATA DESCRIPTION:</u>**

There are 63,000,000 instances and 12 attributes in the training data.
The target variable is ‘label’

which is an ID of respective study protocol condition. The following IDs
are provided: 0 = Transient, 1 = Baseline, 2 = Stress, 3 = Amusement, 4
= Meditation. The data is a multivariate time-series with real-valued
attributes. The preferred data operations associated with the dataset
are classification and regression.

**<u>DATA EXPLORATION:</u>**

The data is in pickle format (.pkl) To be able to perform operations on
the data we use the ‘pickle’ package to convert the data from pickle to
dictionary format. The dictionary contains 3 main entities -

**Subject:** The dataset contains of 15 patients. ‘Subject’ notates the
member’s medical record that

the dataset belongs to.

**Signal**: It describes the body part of which the stress is being
calculated from various attributes.

Signal is being calculated from

> a\. Chest : It has further medical measurements which are used for
> Stress detection
>
> i\. ACC
>
> ii\. ECG iii. EMG iv. EDA v. Temp vi. Resp
>
> b\. Wrist: It has some other medical measurements of the wrist
> included used for Stress
>
> detection
>
> i\. ACC
>
> ii\. BVP iii. EDA iv. Temp

ACC is the accelerometer reading, which are being considered in triaxial
orthogonal directions.

The triaxial ACC readings are different for chest and wrist. The
notations are different to avoid

ambiguity, i.e., for

ACC for Chest : c_ax, c_ay, c_az

ACC for wrist: w_ax, w_ay, w_az

**Label:** The label denotes the target variable for the data mining
operation being performed on

the dataset. It indicates the state of the member in accordance with the
stress level the member

is in, for the reading. Each label has a number code. The Labels are:

> i\. 0 – Baseline
>
> ii\. 1 – Stress
>
> iii\. 2 – Amusement iv. 3 – Meditation

**<u>LABELS DISTRIBUTION IN THE DATA:</u>**

<img src="./clnlwele.png"
style="width:5.41458in;height:5.38861in" />Below is the data
distribution of the counts of labels of the first member

**<u>BALANCING UNBALANCED CHEST DATA:</u>**

The data samples from chest device are 21 times more than data wrist
device samples causing an imbalance, due to which the wrist samples must
be excluded from the dataset. Chest device gives 4255300 samples

<img src="./m5yjjzcx.png" style="width:6.5in;height:6.13125in" />Below
is the distribution visualization of the unbalanced data of chest and
wrist devices:

<img src="./dnipuaft.png"
style="width:5.56944in;height:3.61111in" />

Hence, we convert all the Pickle dictionary records to Dataframe using
Pandas by removing wrist

samples. Now we only have chest device data records.

Once the data is converted to Dataframes we have the following columns
for the data:

“c_ax", "c_ay",
"c_az","c_ecg","c_emg","c_eda","c_temp","c_resp","w_label"

**<u>CALCULATING INTERQUARTILE RANGE & REMOVING OUTLIERS:</u>**

We visualize the data below, by using boxplot:

In descriptive statistics, the interquartile range (IQR) is the measure
of the spread of the middle

half of the data distribution. It is measure of where the bulk of the
values of the data lies.

IQR = Q3 – Q1 , where IQR = Interquartile Range

> Q3 = 3rd Quartile or the 75th Percentile
>
> Q1 = 1st Quartile or the 25th Percentile

<img src="./4g333ibv.png"
style="width:5.22222in;height:3.65278in" />

After performing IQR and removing all the outliers, the data is scaled
down to a measurable

range. The boxplot visualization is as below:

**<u>CORRELATION MATRIX:</u>**

<img src="./zgb1ryog.png"
style="width:5.73611in;height:2.375in" />Below is the correlation matrix
of the data:

<img src="./3ijgnw1u.png"
style="width:5.59722in;height:4.23611in" />

**Heatmap** **of** **the** **data:**

**Interpretations:** From the above correlation matrix and heatmap of
the data, the following interpretations can be made:

> \- c_ax is highly positively correlated with c_az
>
> \- c_eda and c_temp are negatively correlated with each other
>
> \- c_emg and c_ecg are very poorly correlated with the rest of the
> features.

<img src="./4f3mrwk2.png" style="width:6.5in;height:4.66042in" />

**<u>HISTOGRAM:</u>**

Below are the distribution of the data:

**Interpretations:**

We observe that

> \- The IQR is the highest for c_emg & c_ay histograms, followed by
> c_resp and c_ecg graphs
>
> \- There is a high varied distribution for the c_az histogram
>
> \- A bimodal graph is observed with both c_ax, c_eda histograms -
> c_emg shows a uniform distribution graph

<u>DIMENSION REDUCTION AND VARIABLE SELECTION</u>

NORMALIZATION

Z-score normalization is a strategy of normalizing data that avoids this
outlier issue. The formula for Z-score normalization is:

> (Value−μ)/σ

Here, the mean value of the feature is notated by μ, and the standard
deviation of the feature is notated by σ. If a value exactly equals to
the mean of all the values of the feature, it will be normalized to 0.
On the other hand, if it is below the mean, it will be a negative
number, and positive number if above the mean. The size of those
negative and positive numbers is determined by the standard deviation of
the original feature. If the unnormalized data has a large standard
deviation, the normalized values will be closer to 0.

<img src="./egy4chdp.png" style="width:6.5in;height:3.17986in" />After
normalizing the data, the header rows is as follows:

<img src="./n5ke11et.png"
style="width:5.65278in;height:1.70833in" />

PRINCIPAL COMPONENT ANALYSIS (PCA)

Principal component analysis (PCA) is a common method for analyzing huge
datasets with a high number of dimensions/features per observations,
improving data interpretation while retaining the most information, and
enabling the presentation of multidimensional data. Formally, PCA is a
statistical method for lowering a dataset's dimensionality. To do this,
the data are transformed linearly into a new coordinate system, where
(most) of the variance in the data can be expressed with fewer
dimensions than the initial data.

After applying Principal Component Analysis to the dataset, the variance
of the predictor variables is obtained as follows:

<u>DATA MINING MODELS / METHODS</u>

LOGISTIC REGRESSION

Predictive analytics and categorization frequently make use of this kind
of statistical model, also referred to as a logistic regression model.
Based on a given dataset of independent variables, logistic regression
calculates the likelihood that an event will occur, such as voting or
not voting. Given that the result is a probability, the dependent
variable's range is 0 to 1. In logistic regression, the odds—that is,
the probability of success divided by the probability of failure—are
transformed using the logit formula.

ADVANTAGES:

> • The training of logistic regression is very effective and easier to
> implement and analyze. It is also very fast at classifying unknown
> records.
>
> • Althoughit islesslikelytodoso, high-dimensional datasetscancause
> overfittinginlogistic regression. Topreventover-fittinginthese cases,
> one maywant toconsiderregularization (L1 and L2) approaches.
>
> • It can use model coefficients to determine the significance of a
> feature.

DISADVANTAGES:

> • The assumption of linearity between the dependent variable and the
> independent variables is the main drawback of logistic regression.
>
> • Logisticregression hasa lineardecision surface; hence it cannot
> addressnon-linear issues. Real-world situations rarely involve
> linearly separable data.

K-NEAREST NEIGHBOURS

The k-nearest neighbors algorithm, sometimes referred to as KNN or k-NN,
is a supervised learning classifier that employs proximity to produce
classifications or predictions about the grouping of a single data
point. Although it can be applied to classification or regression
issues, it is commonly employed as a classification algorithm because it
relies on the idea that comparable points can be discovered close to one
another.

ADVANTAGES:

> • KNN modeling is very time-efficient in terms of improvisation for a
> random modeling on the available data because itdoes not require
> atraining period as thedata itselfis amodel that will serve as the
> reference for future prediction.
>
> • The only thing that needs to be calculated for KNN is the distance
> between various points using data from variousfeatures,and this
> distance can simply be calculated using distance formulas like
> Euclidian or Manhattan distances.

DISADVANTAGES:

> • Poor performance with unbalanced data – If most of the data the
> model is trained on only contains one label, that label will be highly
> likely to be predicted.
>
> • K value that is optimal — If K is selected wrong, the model will
> either be under- or overfit to the data.

RANDOM FOREST

Random forests or random decision forests are an ensemble learning
method for classification, regression and other tasks that operates by
constructing a multitude of decision trees at training time. For
classification tasks, the output of the random forest is the class
selected by most trees. For regression tasks, the mean or average
prediction of the individual trees is returned.

ADVANTAGES:

> • It reduces overfitting in decision trees and helps to improve the
> accuracy
>
> • It is flexible to both classification and regression problems
>
> • It works well with both categorical and continuous values
>
> • It automates missing values present in the data
>
> • Normalising of data is not required as it uses a rule-based
> approach.

DISADVANTAGES:

> • It requires much computational power as well as resources as it
> builds numerous trees to combine their outputs.
>
> • It also requires much time for training as it combines a lot of
> decision trees to determine the class.
>
> • Due to the ensemble of decision trees, it also suffers
> interpretability and fails to determine the significance of each
> variable.

<u>MODEL PERFORMANCE EVALUATION & VISUALIZATIONS</u>

<img src="./mx3fhkil.png" style="width:6.5in;height:4.98055in" />CONFUSION
MATRIX:

<img src="./h5emyt1p.png"
style="width:5.29514in;height:3.35805in" /><img src="./efd32h2f.png" style="width:6.5in;height:4.83472in" />CLASSIFICATION
SUMMARY REPORT:

**ROC** **Curve:**

An **ROC** **curve** (**receiver** **operating** **characteristic**
**curve**) is a graph showing the performance of a classification model
at all classification thresholds. This curve plots two parameters:

> • True Positive Rate
>
> • False Positive Rate

**True** **Positive** **Rate** (**TPR**) is a synonym for recall and is
therefore defined as follows:

<img src="./k345i0iv.png" style="width:6.5in;height:4.7368in" />**False**
**Positive** **Rate** (**FPR**) is defined as follows:

Interpretations:

From the above we observe that:

> \- The accuracy is 74% for the Logistic Regression Model implemented -
> The F1 score for ‘Stress’ category is the lowest.
>
> \- The categories ‘Baseline’ and ‘Meditation’ have the highest F1
> Score
>
> \- The ROC Curve for the ‘Transient’ category is not ideal, whereas
> for ‘Meditation’ & ‘Baseline’ we obtained the perfect ROC Curve.

LOGISTIC REGRESSION ON PCA WITH 4 PRINCIPAL COMPONENTS

<img src="./p1z535jd.png" style="width:6.5in;height:4.72083in" />After
application of Logistic Regression on PCA Transformed data with 4
Principal Components, the following is the Confusion Matrix:

<img src="./52eydn0o.png"
style="width:5.31167in;height:3.36458in" /><img src="./dbjzzhuc.png" style="width:6.5in;height:4.69653in" />CLASSIFICATION
SUMMARY REPORT:

<img src="./o03la1on.png" style="width:6.5in;height:4.65556in" />

ROC Curve:

Interpretations:

From the above we observe that:

> \- The accuracy is 68.60% for the above implemented model - The F1
> score for ‘Amusement’ category is the lowest.
>
> \- The category ‘Meditation’ continues to have the highest F1 Score in
> this model
>
> \- The ROC Curve for the ‘Transient’ category is not ideal, whereas
> for ‘Meditation’ & ‘Baseline’ we obtained the perfect ROC Curve.

<img src="./hsywfxmm.png" style="width:6.5in;height:4.81805in" /><img src="./j2zps1yd.png"
style="width:4.52986in;height:2.92556in" />LOGISTIC REGRESSION WITH PCA
WITH TWO PRINCIPAL COMPONENTS

<img src="./in3pw0oi.png"
style="width:5.72667in;height:4.14444in" />

<img src="./vd2c01ub.png"
style="width:5.86042in;height:4.26319in" />ROC CURVE:

Interpretations:

From the above we observe that:

> \- The accuracy falls to 54.76% for the above implemented model
>
> \- The F1 score for ‘Baseline’ and ‘Meditation’ categories is the
> lowest. - The category ‘Transient’ has the highest F1 Score in this
> model
>
> \- The ROC Curve yields bad results on all the categories present, and
> is not ideal in either of them.

<img src="./crxvtdid.png" style="width:6.5in;height:3.77847in" />KNN
Model for K = 3

<img src="./gr3jvtg3.png" style="width:6.5in;height:4.14653in" /><img src="./o35eoggo.png"
style="width:5.37917in;height:3.85278in" />

<img src="./telys2nd.png"
style="width:5.56417in;height:4.03403in" />

ROC Curve:

Interpretations:

From the above we observe that:

> \- The KNN Model with K=3, yields good results, with an accuracy of
> 98.19% - The F1 score for ‘Meditation’ category is the lowest.
>
> \- The category ‘Baseline’ has the highest F1 Score in this model
>
> \- The ROC Curve continues to yield great results on all the
> categories present

<img src="./nguwulp5.png" style="width:6.5in;height:3.7375in" /><img src="./gt1pbkta.png" style="width:6.5in;height:4.1875in" />KNN
for K = 5

<img src="./c0dusd2x.png"
style="width:6.06375in;height:4.30556in" />

<img src="./hhxien11.png"
style="width:5.84042in;height:4.2125in" />ROC Curve:

Interpretations:

From the above we observe that:

> \- The KNN Model with K=5, yields good results, with an accuracy of
> 98.05% - The F1 scores for this model are similar to that of K=3
>
> \- The ROC Curve continues to generate ideal results.

<img src="./jcoxgmz5.png" style="width:6.5in;height:3.75347in" />KNN
for K = 11

<img src="./4qdilaby.png" style="width:6.5in;height:4.12292in" /><img src="./5wfy5qmx.png"
style="width:5.61069in;height:4.00486in" />

<img src="./t4h5aelm.png"
style="width:5.59639in;height:4.06458in" />

ROC Curve:

Interpretations:

From the above we observe that:

> \- The KNN Model with K=11, yields good results, with an accuracy of
> 97.6%
>
> \- ‘Stress’ category has the lowest F1 score, while ‘Baseline’
> category has the highest. - The ROC Curve is similar to that of
> previous k values of all KNN models

<img src="./gpfgmt0o.png" style="width:6.5in;height:3.72153in" /><img src="./aocthydx.png" style="width:6.5in;height:4.14792in" />KNN
for K = 21

<img src="./yjmnxbwp.png"
style="width:5.40625in;height:3.85889in" />

<img src="./fmvukjyy.png"
style="width:6.09167in;height:4.4243in" />ROC Curve:

Interpretations:

From the above we observe that:

> \- The KNN Model with K=21 gives an accuracy of 96.95%
>
> \- The F1 score is lowest for ‘Stress’ and highest for ‘Baseline’
> category - The ROC Curve still holds ideal results on all the
> categories present

<img src="./rl42byht.png" style="width:6.5in;height:3.74583in" />KNN
for K = 51

<img src="./xpmcd0fu.png" style="width:6.5in;height:4.22222in" /><img src="./pma5i5no.png"
style="width:5.84861in;height:4.15958in" />

<img src="./rl5ruudy.png"
style="width:5.55417in;height:4.01306in" />

ROC Curve:

Interpretations:

From the above we observe that:

> \- The KNN Model with K=51 gives an accuracy of 95.71%
>
> \- The F1 score is lowest for ‘Stress’ and highest for ‘Baseline’
> categories - The ROC Curve looks ideal following the previous models.

<img src="./glfwlh22.png"
style="width:6.61389in;height:3.78667in" /><img src="./shjkvfy4.png"
style="width:5.9625in;height:3.88389in" />KNN for K = 100

<img src="./jvugvm4t.png" style="width:6.5in;height:4.63958in" />

<img src="./xowx4uyj.png" style="width:6.5in;height:4.7368in" />

ROC Curve:

Interpretations:

From the above we observe that:

> \- The KNN Model with K=100 , slightly drops accuracy but still
> considered as good results, with an accuracy of 94.54%
>
> \- The ‘Stress’ category has the lowest F1 Score, while the Baseline
> and Meditation have the highest and subsequent F1 scores respectively
>
> \- The ROC Curve continues togenerate similar results tothat of
> previous k values of all KNN models

<img src="./153uilki.png" style="width:6.5in;height:4.05069in" />

FINDING THE RIGHT K VALUE

As depicted from the graph, the error rate between k=3 & 5 values is the
lowest, and the difference in error rate increases as k value is
increased.

<img src="./yrecfayz.png"
style="width:5.60764in;height:3.20347in" />RANDOM FOREST

<img src="./js3q2jlf.png" style="width:6.5in;height:4.25069in" /><img src="./gvxz1b45.png"
style="width:5.9625in;height:4.26292in" />

<img src="./3yb225ex.png" style="width:6.5in;height:4.69653in" />

ROC Curve:

Interpretations:

From the above we observe that:

> \- The ROC Model shows the best results so far, with an accuracy of
> 98.99%
>
> \- The ‘Meditation’ category has the lowest F1 Score, while the
> Baseline has the highest score
>
> \- The ROC Curve also gives the perfect results on models so far
> implemented

<img src="./zdbo2tai.png" style="width:6.5in;height:3.7375in" /><img src="./ycd2zdqh.png" style="width:6.5in;height:4.72083in" />RANDOM
FOREST ON PCA

<img src="./1txb2glw.png"
style="width:6.1475in;height:4.04444in" />

> <img src="./ama0tjv0.png"
> style="width:6.02722in;height:4.34722in" />ROC Curve :

Interpretations:

From the above we observe that:

> \- The Random Forest Model on PCA also gives great results with an
> accuracy of 96.67%
>
> \- The ‘Transient’ Category has the highest F1 score, while the
> ‘Baseline’ category has the lowest
>
> \- The ROC Curve also looks relatively ideal across the categories.

<img src="./eqvlvaaz.png" style="width:6.5in;height:4.72083in" /><u>F1
SCORE ACROSS ALL CATEGORIES:</u>

<img src="./jvsrtihe.png" style="width:6.5in;height:5.41111in" />

MODEL IMPLEMENTATION INTERPRETATION:

After implementing the Logistic Regression, KNN Model on different k
values and Random Forest Models on the dataset, different parameters for
model’s selection are calculated. It is observed that while the accuracy
and other parameter scores of Logistic Regression aren’t relatively
great, KNN yields better results on the dataset and Random Forest Model
generates the best results on the dataset with an accuracy of 98.99%.
The KNN Model is the subsequent best model with k=3 with an accuracy
over 98%

Implementation of PCA on Logistic Regression generated bad results with
accuracy rates of 68% and 58%. While the PCA implementation worked great
on Random Forest giving an accuracy of 96% This means PCA worked best on
Random Forest but couldn’t improve Logistic Regression model results
relatively. Based on the results, the implementation of Random Forest
model

generates best results on the dataset once trained, relatively as
compared to other models. Logistic Regression is a bad model to
implement.

REFERENCES:

> 1\. Dataset:
> [<u>https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Dete</u>](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
> [<u>ction%29</u>](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
>
> 2\. Relevant Papers:
> [<u>https://dl.acm.org/doi/10.1145/3242969.3242985</u>](https://dl.acm.org/doi/10.1145/3242969.3242985)
