#!/usr/bin/python
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display
import os

#%matplotlib inline
sns.set(style="whitegrid", color_codes=True)


from pprint import pprint
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit

from sklearn.pipeline import Pipeline, make_pipeline

# import tools
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

# set work directory
os.chdir("/Users/linzhu/Documents/mooc/dand/dand_p5_enron_email")

### Task 1: Data Exploration
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# transfer the dictionary into dataframe
data = pd.DataFrame(data_dict).T
print "The original dataset has {} rows and {} columns".format(data.shape[0], data.shape[1])



#### Task 2: Remove outliers

print data.loc['THE TRAVEL AGENCY IN THE PARK']
print "##########"
print data.loc['TOTAL']  
## remove "TOTAL"
data.drop('TOTAL', inplace = True)
data.shape
print "The new dataset has {} rows and {} columns".format(data.shape[0], data.shape[1])

## find data with wrong sign and correct them
def print_sign(feature, sign, correction = False):
    for index in data.index:
        v = data.loc[index, feature]
        if sign == "positive":
            if v != "NaN" and v > 0:
                print "'{}' shows feature '{}' as {} with wrong sign, \
                should be 'negative'".format(index,feature, v)
                if correction:
                    data.loc[index, feature] = - v
                    print "'{}' has correct value {} in '{}'".format\
                    (index, data.loc[index, feature], feature)
                    print ""
                    print ""
                    
            
        elif sign == "negative":
            if v != "NaN" and v < 0:
                print "'{}' shows feature '{}' as {} with wrong sign, \
                should be 'positive'".format(index,feature, v) 
                if correction:
                    data.loc[index, feature] = - v
                    print "'{}' has correct value {} in '{}'".format\
                    (index, data.loc[index, feature], feature)
                    print ""
                    print ""

# generate positive and negative list
negative_list = ['deferred_income', 'restricted_stock_deferred']
positive_list = data.columns.values.tolist()
print type(positive_list)
positive_list.remove('deferred_income')
positive_list.remove('restricted_stock_deferred')

# check wrong sign and correct them
for feature in negative_list:
    print_sign(feature, "positive", correction = True)

for feature in positive_list:
    print_sign(feature, "negative", correction = True)

## Allocation POI/non-POI
poi = data[data.poi == True]
non_poi = data[data.poi == False]
print "################"
print "The original dataset contains:"
print "#{} poi".format(len(poi.index))
print "#{} non-poi".format(len(non_poi.index))

## Check Missing Values
# data.describe()
feature = data.columns.values
print feature

############# 
# Missing value allocation
total_num = len(data['poi'])
poi_num = len(data[data.poi == True]['poi'])
nonpoi_num = len(data[data.poi == False]['poi'])
print total_num, poi_num, nonpoi_num

total_null = []
poi_null = []
nonpoi_null = []

total_r = []
poi_r = []
nonpoi_r = []

for col in data.columns:
    total_null.append(len(data[data[col] == 'NaN']))
    poi_null.append(len(data[data[col] == 'NaN'][data.poi == True]))
    nonpoi_null.append(len(data[data[col] == 'NaN'][data.poi == False]))

    t_r = len(data[data[col] == 'NaN']) / float(total_num)
    p_r = len(data[data[col] == 'NaN'][data.poi == True]) / float(poi_num)
    np_r = len(data[data[col] == 'NaN'][data.poi == False]) / \
    float(nonpoi_num)
    total_r.append("{0:.4f}".format(t_r))
    poi_r.append("{0:.4f}".format(p_r))
    nonpoi_r.append("{0:.4f}".format(np_r))

count_null = pd.DataFrame(dict(feature = feature, total_null = total_null,\
                               poi_null = poi_null, nonpoi_null = \
                               nonpoi_null, total_r = total_r, poi_r = \
                               poi_r, nonpoi_r = nonpoi_r))

def null_filter(perct):
    '''
    inputs the satisfied maximum percentage of missing value out of total 
    value counts for a specific feature,
    returns names of satisfied features with missing value less than the 
    input percentage
    '''
    size = len(data) * perct
    l = []
    index = []
    df = pd.DataFrame()
    for i in range(0, len(count_null)):
        if count_null.loc[i, "total_null"] <= size:
            l.append(count_null.loc[i, "feature"])
            index.append(i)
    count = len(l)
    print "{} features in total, including: {}".format(count, l)
    
    return count_null.loc[index, :]
# print features with more than 60% valid value
sub_dict = null_filter(0.4)
sub_dict.sort("total_null")

# Plot out those features
#Set general plot properties
sns.set_style("white")
sns.set_context({"figure.figsize": (24, 10)})

#Plot 1 - background - "total" (top) series
sns.barplot(y = count_null.feature, x = count_null.total_null, color = "blue")
# sns.barplot(x = count_null.feature, y = count_null.total_null, color = "red")

#Plot 2 - overlay - "bottom" series
bottom_plot = sns.barplot(y = count_null.feature, x = count_null.poi_null, color = "red")
# bottom_plot = sns.barplot(x = count_null.feature, y = count_null.poi_null, color = "#0000A3", )


topbar = plt.Rectangle((0,0),1,1,fc='blue', edgecolor = 'none')
bottombar = plt.Rectangle((0,0),1,1,fc="red",  edgecolor = 'none')
l = plt.legend([bottombar, topbar], ['POI with Missing Value','Non-POI with Missing Value'], loc= 1, ncol = 2, prop={'size':16})
l.draw_frame(False)

#Optional code - Make plot look nicer
sns.despine(left=True)
bottom_plot.set_xlabel("Number of Missing Values")
bottom_plot.set_ylabel("Features")
bottom_plot.set_title("Comparison of 'NaN' Counts Between Whole Group and POI")

#Set fonts to consistent 16pt size
for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
    item.set_fontsize(20)




#### Task 3: Create new feature(s)
#### Store to my_dataset for easy export below.
# feature selected from 40% missing value filter. 
feature = ['poi', 'salary', 'email_address', 'exercised_stock_options', 'expenses', 'other', \
           'restricted_stock', 'total_payments', 'total_stock_value']

# define indicator function
def add_indicator(indicator_feature):
    for feature in indicator_feature:
        col_name = feature + str('_indicator')
        if col_name in data.columns.tolist():
            pass
        else:
            col = []
            for index in data.index:
                v = data.loc[index, feature]
                if v == "NaN":
                    data.loc[index, col_name] = 0
                else:
                    data.loc[index, col_name] = 1

indicator_feature = ['salary', 'email_address', 'exercised_stock_options', 'expenses', \
                'other', 'restricted_stock', 'total_payments', 'total_stock_value']
add_indicator(indicator_feature)


### Task 4: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# drop "email_address"
data = data.drop("email_address", axis = 1)
print data.head(2)

# convert the dataframe back into dict
my_dataset = data.T.to_dict()
features_list = data.columns.tolist()
print len(features_list)
print features_list

# reoder and put "poi" to the first place in 'features_list'
features_list.remove("poi") 
features_list.insert(0, "poi")
print features_list

# get labels and features
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data) # labels and features are both list

# utilize decision tree to rank features by their importance
# the reason why using decison tree is because they are not 

clf = DecisionTreeClassifier(random_state = 0) # fit using defult value
clf.fit(features, labels)

feature_importances = clf.feature_importances_
sorted_id = (-np.array(feature_importances)).argsort()
print "Rank of features"
j = 1
for i in sorted_id:
    print "Ranking place {}, value {:4f}: {}".\
    format(j,feature_importances[i], features_list[i+1])
    j += 1
    

### Task 5: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# select top 8 features
features_list = ["poi", "exercised_stock_options", "bonus", \
                 "total_payments", "from_messages",'long_term_incentive', \
                 "shared_receipt_with_poi", 'expenses']
# , 'to_messages','other', 'restricted_stock', 'from_this_person_to_poi'
print len(features_list)
print features_list

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# split data into training and testing set
X_train, X_test, y_train, y_test = \
train_test_split(features, labels, test_size = 0.2,random_state = 0)

# build pipeline for four estimator, train, predict and compare scores
scaler = StandardScaler()
kbest = SelectKBest(f_classif, k = 2)
gnb = GaussianNB()
svc = SVC()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

pipe1 = Pipeline([("scaler",scaler), ('kbest', kbest), ('gnb', gnb)])
pipe2 = Pipeline([("scaler",scaler), ('kbest', kbest), ('svc', svc)])
pipe3 = Pipeline([("scaler",scaler), ('kbest', kbest), ('dt', dt)])
pipe4 = Pipeline([("scaler",scaler), ('kbest', kbest), ('knn', knn)])

# fit and predict utilizing Gaussian Naive Bayes
pipe1.fit(X_train, y_train)
y_pred1 = pipe1.predict(X_test)
print "Classification report for Gaussian Naive Bayes:"
# print(classification_report(y_test, y_pred1))
test_classifier(gnb, my_dataset, features_list)

# fit and predict utilizing SVC
pipe2.fit(X_train, y_train)
y_pred2 = pipe2.predict(X_test)
print "Classification report for SVC:"
# print(classification_report(y_test, y_pred2))
test_classifier(svc, my_dataset, features_list)

# fit and predict utilizing Decision Tree
pipe3.fit(X_train, y_train)
y_pred3 = pipe3.predict(X_test)
print "Classification report for Decision Tree:"
# print(classification_report(y_test, y_pred3))
test_classifier(dt, my_dataset, features_list)

# fit and predict utilizing KNN
pipe4.fit(X_train, y_train)
y_pred4 = pipe4.predict(X_test)
print "Classification report for KNN:"
# print(classification_report(y_test, y_pred4))
test_classifier(knn, my_dataset, features_list)





### Task 6: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## Tuning Parameters
pipe = Pipeline([('scaler', scaler), ('kbest', kbest), ('dt', dt)])

params = {
    'dt__min_samples_split': [2, 5, 10, 20, 40],
#     'dt__min_samples_split':[2, 3, 4, 5, 6, 7, 8],
    'dt__max_depth':[None, 2, 4, 6, 8, 10, 15, 20],
    'kbest__k': [1, 2, 3, 4, 5, 6, 7, 8],
    'kbest__score_func': [f_classif, chi2],
    }

# split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, \
                                                    random_state = 0)

# make stratifiedShuffleSplit iterator for cross-validation in GridSearchCV
sss = StratifiedShuffleSplit(y_train, n_iter = 20, test_size = 0.2, random_state = 0)

# Apply GridSearchCV and run cross_validation
clf = GridSearchCV(pipe,
                   param_grid = params,
                   scoring = 'f1',
                   n_jobs = 1, 
                   cv = sss, 
                   verbose = 1, 
                   error_score = 0
                  )

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print '\n',"Classification performance report for tuned Gaussian Naive Bayes:"
print (classification_report(y_test,y_pred))
print 'Best estimator:'
print clf.best_estimator_
clf = clf.best_estimator_

test_classifier(clf, my_dataset, features_list)


### Task 7: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)