# Gauss Naive Bayes In Python

## Overview
Creating a classifier using the Normal PDF (Probability Distribution Function) and the Joint PDF in python from scratch.

#### Normal PDF:
![Alt text](img/normal_distribution.svg "Optional Title")

See [Normal Distribution (Wikipedia)](https://en.wikipedia.org/wiki/Normal_distribution) definition.

`x` is the target class and the variable that we will predict.
The Normal Distribution will help determine the likelihood of `x` occuring for each feature. In other words for each column of our dataset, the Normal Distribution will calculate the probability of `x` occuring. 

#### Joint PDF:
![Alt text](img/joint_pdf.svg "Optional Title")

See [Joint PDF (Wikipedia )](https://en.wikipedia.org/wiki/Joint_probability_distribution) definition.

The Joint PDF simply joins all of the PDFs. In our case it's the Normal Distribution PDF. The results from the PDF, for each feature, are all multiplied and the result is the Joint PDF.
## Getting Started

Git clone the repo to use the code 
```
git clone https://github.com/odubno/naive_bayes.git
```

### Prerequisites


Every function is created from scratch.
However, instead of having to download the data, we're using a quick api call to get the csv.

```
pip install requests
```

### Step by Step

#### Data
- Data is comma separated.
- Each row repesents an individual data point.
- Each column represents a feature.
- Last column represents the target class for each row.
```
6,148,72,1
1,85,66,0
8,183,64,1
1,89,66,0
0,137,40,1
```

#### Splitting Data
```python
def split_data(self, data, weight):
    """
    :param data: original data set
    :param weight: percentage of data used for training
    :return:
    append rows to train while removing those same rows from data
    """
    train_size = int(len(data) * weight)
    train_set = []
    for i in range(train_size):
        index = random.randrange(len(data))
        train_set.append(data[index])
        data.pop(index)
    return [train_set, data]
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

* **Oleh Dubno** - *algorithms and python code* - [odubno](http://odubno.github.io/)
* **Danny Argov** - *algorithms and idea generation*

See also the list of [contributors](https://github.com/odubno/naive_bayes/graphs/contributors) who participated in this project.


## Acknowledgments

* Hat tip to Dr. Jason Brownlee, who wrote a blog of [How To Implement Naive Bayes From Scratch in Python](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/). 
Much of the logic here comes from his post. 
* Inspiration:
Project for Columbia Probability and Statistics course - Prof. Banu Baydil
