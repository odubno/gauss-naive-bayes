# Gauss Naive Bayes In Python



## Table of Contents
  - [Overview](#overview)
    - [Iris Data Set](#iris-data-set)
    - [Bayes Theorem](#bayes-theorem)
    - [Normal Probability Density Function](#normal-pdf)
    - [Joint Probability Density Function](#joint-pdf)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [State By Step](#step-by-step)
      - [Data](#data)
      
## Overview 
We will be using Naive Bayes and the Gaussian Distribution (Normal Distribution) to build a classifier in Python from scratch.

The Gauss Naive Bayes Classifier will run on four classic data sets:

* [iris](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
* [diabetes](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)
* [redwine](http://archive.ics.ucimachine-learning-databases/wine-quality/winequality-red.csv)
* [adult](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)


Here we'll be working with just the [iris](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) data set.

#### Iris Data Set:

The Iris data set is a classic and is widely used when explaining classification models. 
The data set has 4 independent variables and 1 dependent variable that has 3 different classes.

The first 4 columns are the independent variables (features).                                      
The 5th column is the dependent variable (class).

1. *sepal length* (cm)
2. *sepal width* (cm) 
3. *petal length* (cm) 
4. *petal width* (cm) 
5. classes: 
    * *Iris Setosa*, 
    * *Iris Versicolour*
    * *Iris Virginica*

- 5 row sample from the Iris data
- The first 4 columns represent the **features** (*sepal length*, *sepal width*, *petal length*, *petal width*)
- The last column represents the **class** for each row. (*Setosa*, *Versicolour*, *Virginica*)

| sepal length  | sepal width | petal length | petal width | class |
| :-----------: |:-----------:| :----------: | :----------:| :----:|
| 5.1 | 3.5 | 1.4 | 0.2| Iris-setosa | 
| 4.9 | 3.0 | 1.4 | 0.2| Iris-setosa |
| 7.0 | 3.2 | 4.7 | 1.4| Iris-versicolor |
| 6.3 | 2.8 | 5.1 | 1.5| Iris-virginica |
| 6.4 | 3.2 | 4.5 | 1.5| Iris-versicolor |

#### Bayes Theorem:
![Bayes](img/bayes_exp.JPG "Bayes" )

**Class Prior Probability:** 
* This is our Prior Belief

**Likelihood:**
* We are using the [Normal Distribution (Gauss)](#normal-pdf) to calculate this. Hence, the name Gauss Naive Bayes.

**Predictor Prior Probability:**
* Marginal probability of how probable the new evidence is under all possible hypothesis. Most Naive Bayes Classifiers do not calculate this. The results do not change or change very little. Though we do calculate it here.


#### Normal PDF:
![Normal Distribution](img/normal_distribution.svg "Normal Distribution" )

See [Normal Distribution (Wikipedia)](https://en.wikipedia.org/wiki/Normal_distribution) definition.

The Normal Distribution will help determine the likelihood of a *class* occuring for each feature. In other words for each column of our dataset, the Normal Distribution will calculate the likelihood of that *class* occuring. 

#### Joint PDF:
![Alt text](img/joint_pdf.svg "Optional Title")

See [Joint PDF (Wikipedia )](https://en.wikipedia.org/wiki/Joint_probability_distribution) definition.

The Joint PDF is the product of all PDFs. In our case, the product of all Normal Distribution PDFs. Multiplying all of the PDFs gives us the likelihood.


## Prepare Data

Building the Naive Bayes Classifier. 

### Prerequisites


Every function is created from scratch.
However, instead of having to download the data, we're using a quick api call to get the csv.

```
$ pip install requests
```

### Skeleton

Create the skeleton: 
- import the necessary libraries and create the class.

```python
# -*- coding: utf-8 -*-
from collections import defaultdict
from math import pi
from math import e
import requests
import random
import csv
import re

class GaussNB:

    def __init__(self):
        pass
        
        
def main():
    print "Here we will handle class methods."
    
    
if __name__ == '__main__':
    main()
```
Execute in terminal:
```
$ python gauss_nb.py
```


###### Output:
```
Here we will handle class methods.
```

#### Load CSV

Writing the method to read in the raw data.


```python
class GaussNB:

    def __init__(self):
        pass
        
    def load_csv(self, data, header=False):
        """
        :param data: raw comma seperated file
        :param header: remove header if it exists
        :return:
        Load and convert each string of data into a float
        """
        lines = csv.reader(data.splitlines())
        dataset = list(lines)
        if header:
            # remove header
            dataset = dataset[1:]
        for i in range(len(dataset)):
            dataset[i] = [float(x) if re.search('\d', x) else x for x in dataset[i]]
        return dataset
        
def main():
    nb = GaussNB()
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = requests.get(url).content
    data = nb.load_csv(data, header=True)
    print data[:3] # first 3 rows
    
    
if __name__ == '__main__':
    main()
```

###### Output:
```
[[4.9, 3.0, 1.4, 0.2, 'Iris-setosa'], [4.7, 3.2, 1.3, 0.2, 'Iris-setosa'], [4.6, 3.1, 1.5, 0.2, 'Iris-setosa']]
```

#### Split Data
Splitting data into train and test set.
The weight will determine how much of the data will be training data.

```python
class GaussNB:
    .
    .
    .
    def split_data(self, data, weight):
        """
        :param data:
        :param weight: indicates the percentage of rows that'll be used for training
        :return:
        Randomly selects rows for training according to the weight and uses the rest of the rows for testing.
        """
        train_size = int(len(data) * weight)
        train_set = []
        for i in range(train_size):
            index = random.randrange(len(data))
            train_set.append(data[index])
            data.pop(index)
        return [train_set, data]


def main():
    nb = GaussNB()
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = requests.get(url).content
    data = nb.load_csv(data, header=True)
    train_list, test_list = nb.split_data(data, weight=.67)
    print "Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list))

if __name__ == '__main__':
    main()
```


#### Group Data

Grouping data by class. This method will map each class to it's respective rows of data.

e.g. (This is just a sample. Using the [table](#iris-data-set) from above.)
```python
{
       'Iris-virginica': [
        [6.3, 2.8, 5.1, 1.5],
    ], 'Iris-setosa': [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
    ], 'Iris-versicolor': [
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
    ]
}
```

```python
class GaussNB:
    .
    .
    .
    def group_by_class(self, data, target):
        """
        :param data: Training set. Lists of events (rows) in a list
        :param target: Index for the target column. Usually the last index in the list
        :return:
        Mapping each target to a list of it's features
        """
        target_map = defaultdict(list)
        for index in range(len(data)):
            features = data[index]
            if not features:
                continue
            x = features[target]
            target_map[x].append(features[:-1])  # designating the last column as the class column
        print 'Identified %s different target classes: %s' % (len(target_map.keys()), target_map.keys())
        return dict(target_map)

def main():
    nb = GaussNB()
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = requests.get(url).content
    data = nb.load_csv(data, header=True)
    train_list, test_list = nb.split_data(data, weight=.67)
    print "Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list))
    group = nb.group_by_class(data, -1)
    print "Grouped into %s classes: %s" % (len(group.keys()), group.keys())

if __name__ == '__main__':
    main()
```

###### Output:
`Grouped into 3 classes: ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']`

### Summarize Data

#### Mean

```python
class GaussNB:
    .
    . 
    . 
    def mean(self, numbers):
        """
        :param numbers: list of numbers
        :return: 
        """
        result = sum(numbers) / float(len(numbers))
        return result

def main():
    nb = GaussNB()
    print "Mean: %s" % nb.mean([5.9, 3.0, 5.1, 1.8])

if __name__ == '__main__':
    main()
```
###### Output:
```
Mean: 3.95
```

#### Standard Deviation

```python
class GaussNB:
    . 
    . 
    . 
    def stdev(self, numbers):
        """
        :param numbers: list of numbers
        :return:
        Calculate the standard deviation for a list of numbers.
        """
        avg = self.mean(numbers)
        squared_diff_list = []
        for num in numbers:
            squared_diff = (num - avg) ** 2
            squared_diff_list.append(squared_diff)
        squared_diff_sum = sum(squared_diff_list)
        sample_n = float(len(numbers) - 1)
        var = squared_diff_sum / sample_n
        return var ** .5

def main():
    nb = GaussNB()
    print "Standard Deviation: %s" % nb.stdev([5.9, 3.0, 5.1, 1.8])

if __name__ == '__main__':
    main()
```
###### Output:
```
Standard Deviation: 1.88414436814
```

#### Summary
Returns a the mean and standard deviation for each column of the data set,

```python
class GaussNB:
    . 
    . 
    . 
    def summarize(self, data):
        """
        :param data: lists of events (rows) in a list
        :return:
        Use zip to line up each feature into a single column across multiple lists.
        yield the mean and the stdev for each feature
        """
        for attributes in zip(*data):
            yield {
                'stdev': self.stdev(attributes),
                'mean': self.mean(attributes)
            }

def main():
    nb = GaussNB()
    data = [
        [5.9, 3.0, 5.1, 1.8], 
        [5.1, 3.5, 1.4, 0.2]
    ]
    print "Feature Summary: %s" % [i for i in nb.summarize(data)]

if __name__ == '__main__':
    main()
```
###### Output:
```
Feature Summary: 
[
    {'mean': 5.5, 'stdev': 0.5656854249492386}, # sepal length 
    {'mean': 3.25, 'stdev': 0.3535533905932738}, # sepal width
    {'mean': 3.25, 'stdev': 2.6162950903902256}, # petal length
    {'mean': 1.0, 'stdev': 1.1313708498984762} # petal width
]
```

### Calculate Prior Probability
Calculating the prior probability for each class.
Prior probability is the probability of the each class occuring.

```python
class GaussNB:
    . 
    . 
    . 
    def prior_prob(self, group, target, data):
        """
        :return:
        The probability of each target class
        """
        total = float(len(data))
        result = len(group[target]) / total
        return result

def main():
    nb = GaussNB()
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = requests.get(url).content
    data = nb.load_csv(data, header=True)
    train_list, test_list = nb.split_data(data, weight=.67)
    print "Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list))
    group = nb.group_by_class(data, -1)  # designating the last column as the class column
    print "Grouped into %s classes: %s" % (len(group.keys()), group.keys())
    for target_class in ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']:
        prior_prob = nb.prior_prob(group, target_class, data)
        print 'P(%s): %s' % (target_class, prior_prob)

if __name__ == '__main__':
    main()
```
###### Output:
```
Using 100 rows for training and 50 rows for testing
Grouped into 3 classes: ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
P(Iris-virginica): 0.38
P(Iris-setosa): 0.3
P(Iris-versicolor): 0.32
```

### Train
Putting the previous methods together to determine the prior for each class and the (mean, standard deviation) combination for each feature of each class.
```python
class GaussNB:
    . 
    . 
    . 
    def train(self, train_list, target):
        """
        :param data:
        :param target: target class
        :return:
        For each target:
            1. yield prior: the probability of each class. P(class) eg P(Iris-virginica)
            2. yield summary: list of {'mean': 0.0, 'stdev': 0.0}
        """
        group = self.group_by_class(train_list, target)
        self.summaries = {}
        for target, features in group.iteritems():
            self.summaries[target] = {
                'prior': self.prior_prob(group, target, train_list),
                'summary': [i for i in self.summarize(features)],
            }
        return self.summaries

def main():
    nb = GaussNB()
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = requests.get(url).content
    data = nb.load_csv(data, header=True)
    train_list, test_list = nb.split_data(data, weight=.67)
    print "Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list))
    group = nb.group_by_class(data, -1)  # designating the last column as the class column
    print "Grouped into %s classes: %s" % (len(group.keys()), group.keys())
    print nb.train(train_list, -1)

if __name__ == '__main__':
    main()
```
###### Output:
```
Using 100 rows for training and 50 rows for testing
Grouped into 3 classes: ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
{'Iris-setosa': {'prior': 0.3,
  'summary': [{'mean': 4.980000000000001, 'stdev': 0.34680810554104063}, # sepal length 
   {'mean': 3.406666666666667, 'stdev': 0.3016430104397023}, # sepal width
   {'mean': 1.496666666666667, 'stdev': 0.20254132542705236}, # petal length
   {'mean': 0.24333333333333343, 'stdev': 0.12228664272317624}]}, # petal width
 'Iris-versicolor': {'prior': 0.31,
  'summary': [{'mean': 5.96774193548387, 'stdev': 0.4430102307127106},
   {'mean': 2.7903225806451615, 'stdev': 0.28560443356698495},
   {'mean': 4.303225806451613, 'stdev': 0.41990782398659987},
   {'mean': 1.3451612903225807, 'stdev': 0.17289439874755796}]},
 'Iris-virginica': {'prior': 0.39,
  'summary': [{'mean': 6.679487179487178, 'stdev': 0.585877428882027},
   {'mean': 3.002564102564103, 'stdev': 0.34602036712733625},
   {'mean': 5.643589743589742, 'stdev': 0.5215336048086158},
   {'mean': 2.0487179487179477, 'stdev': 0.2927831916298213}]}}

```

### Normal PDF
The Normal Distribution will determine the likelihood of each feature for the test set.

E.g.

As a quick example below, we're using Normal Distribution to determine the likelihood that 5 will occur given the mean of 4.98 and the standard deviation of 0.35.

FYI, we're "testing" 5 against Iris-setosa: {'mean': 4.980000000000001, 'stdev': 0.34680810554104063} of the sepal width.



```python
class GaussNB:
    . 
    . 
    .
    def normal_pdf(self, x, mean, stdev):
        """
        :param x: a variable
        :param mean: µ - the expected value or average from M samples
        :param stdev: σ - standard deviation
        :return: Gaussian (Normal) Density function.
        N(x; µ, σ) = (1 / 2πσ) * (e ^ (x–µ)^2/-2σ^2
        """
        variance = stdev ** 2
        exp_squared_diff = (x - mean) ** 2
        exp_power = -exp_squared_diff / (2 * variance)
        exponent = e ** exp_power
        denominator = ((2 * pi) ** .5) * stdev
        pdf = exponent / denominator
        return pdf

def main():
    nb = GaussNB()
    normal_pdf = nb.normal_pdf(5, 4.98, 0.35)
    print normal_pdf

if __name__ == '__main__':
    main()
```
###### Output:
```
1.13797564994
```

### Marginal PDF

```python
class GaussNB:
    . 
    . 
    . 
    def marginal_pdf(self, pdfs):
        """
        :param pdfs: list of probability densities for each feature
        :return:
        Marginal Probability Density Function (Predictor Prior Probability)
        Summing up the product of P(class) prior probability and the probability density of each feature P(feature | class)

        marginal pdf =
          P(setosa) * P(sepal length | setosa) + P(versicolour) * P(sepal length | versicolour) + P(virginica) * P(sepal length | verginica)
        + P(setosa) * P(sepal width | setosa) + P(versicolour) * P(sepal width | versicolour) + P(virginica) * P(sepal width | verginica)
        + P(setosa) * P(petal length | setosa) + P(versicolour) * P(petal length | versicolour) + P(virginica) * P(petal length | verginica)
        + P(setosa) * P(petal length | setosa) + P(versicolour) * P(petal length | versicolour) + P(virginica) * P(petal length | verginica)
        """
        predictors = []
        for target, features in self.summaries.iteritems():
            prior = features['prior']
            for index in range(len(pdfs)):
                normal_pdf = pdfs[index]
                predictors.append(prior * normal_pdf)
        marginal_pdf = sum(predictors)
        return marginal_pdf

def main():
    nb = GaussNB()
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = requests.get(url).content
    data = nb.load_csv(data, header=True)
    train_list, test_list = nb.split_data(data, weight=.67)
    print "Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list))
    group = nb.group_by_class(data, -1)  # designating the last column as the class column
    print "Grouped into %s classes: %s" % (len(group.keys()), group.keys())
    nb.train(train_list, -1)
    marginal_pdf = nb.marginal_pdf(pdfs=[0.02, 0.37, 3.44e-12, 4.35e-09])
    print marginal_pdf

if __name__ == '__main__':
    main()
```
###### Output:
```
Using 100 rows for training and 50 rows for testing
Grouped into 3 classes: ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
0.38610000431
```

#### Train
Calculate the mu and variance of features for each target class.

#### Test
Using the the Normal Distribution, calculate the PDF for all test features of each target class using N(x; mu, variance).
Calculate the Joint PDF of each row, by multiplying pdf results together.
Choose the highest joint pdf.


### Installing

Once cloned, running the program is as simple as calling 
```bazaar
python naive_bayes.py
```


End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Authors

* **Oleh Dubno** - [github.odubno](http://odubno.github.io/)
* **Danny Argov**

See also the list of [contributors](https://github.com/odubno/naive_bayes/graphs/contributors) who participated in this project.


## Acknowledgments


#### Sources:
Hat tip to the authors that made this code possible: 

| Author                  | URL           |
| -------------           |:-------------:|
| Dr. Jason Brownlee      | [How To Implement Naive Bayes From Scratch in Python](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/) |
| Chris Albon             | [Naive Bayes Classifier From Scratch](https://chrisalbon.com/machine-learning/naive_bayes_classifier_from_scratch.html) |
| Sunil Ray               | [6 Easy Steps to Learn Naive Bayes Algorithm](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/) |
| Rahul Saxena            | [How The Naive Bayes Classifier Works In Machine Learning](http://dataaspirant.com/2017/02/06/naive-bayes-classifier-machine-learning/) |
| Data Source             | [UCI Machine Learning](http://archive.ics.uci.edu/ml/index.php) |

#### Inspiration:  
Project for Columbia Probability and Statistics course - Prof. Banu Baydil
