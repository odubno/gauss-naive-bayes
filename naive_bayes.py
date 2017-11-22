# -*- coding: utf-8 -*-
import random
import csv
from collections import defaultdict
from math import e
from math import pi
import requests
import re


class GaussNB:

    def __init__(self, normalize=False, standardize=False):
        self.summaries = {}
        self.normalize = normalize
        self.standardize = standardize

    def load_csv(self, data, clean='', header=False, rows=0, normalize=False, delimeter=','):
        """
        :param data:
        :return:
        Load and convert each string of data into float
        """
        lines = csv.reader(data.splitlines(), delimiter=delimeter)
        dataset = list(lines)
        if header:
            # remove header
            dataset = dataset[1:]
        if rows:
            dataset = dataset[:rows]
        if clean in ['adult']:
            for i in range(len(dataset)):
                if not dataset[i]:
                    # skipping empty rows
                    continue
                sex = dataset[i][9].lower().strip()
                dataset[i] = [float(re.search('\d+', x).group(0)) for x in dataset[i] if re.search('\d+', x)]
                dataset[i].append(sex)
        elif clean in ['iris', 'diabetes', 'redwine']:
            for i in range(len(dataset)):
                dataset[i] = [float(x) if re.search('\d', x) else x for x in dataset[i]]
        else:
            print 'Add dataset.'
            return None
        return dataset

    def normalize_data(self, data):
        minimum = min(data)
        maximum = max(data)
        min_max = minimum - maximum
        result = [abs((i - minimum)/(min_max)) for i in data]
        return result

    def standardize_data(self, data):
        stdev = self.stdev(data)
        avg = self.mean(data)
        result = [(i - avg)/ stdev for i in data]
        return result

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

    def mean(self, numbers):
        result = sum(numbers) / float(len(numbers))
        return result

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

    def group_by_target(self, data, target):
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
            target_map[x].append(features[:-1])
        print 'Identified %s different target classes: %s' % (len(target_map.keys()), target_map.keys())
        return dict(target_map)

    def summarize(self, data):
        """
        :param data: lists of events (rows) in a list
        :return:
        Use zip to line up each feature into a single column across multiple lists.
        yield the mean and the stdev for each feature
        """
        for attributes in zip(*data):
            if self.normalize:
                attributes = self.normalize_data(attributes)
            if self.standardize:
                attributes = self.standardize_data(attributes)
            yield {
                'stdev': self.stdev(attributes),
                'mean': self.mean(attributes)
            }

    def train(self, data, x_target):
        """
        :param data:
        :param x_target: dependent variable/ predicted variable
        :return:
        For each target:
            1. yield prior_prob: the probability of that target. eg P(Iris-virginica)
            2. yield summary: list of [{'mean': 0.0, 'stdev': 0.0}, ...]
        """
        class_feature_map = self.group_by_target(data, x_target)
        self.summaries = {}
        for target, features in class_feature_map.iteritems():
            self.summaries[target] = {
                'prior_prob': self.probability(class_feature_map, target, data),
                'summary': [i for i in self.summarize(features)],
            }
        return self.summaries

    def probability(self, class_feature_map, target, data):
        """
        :return:
        The probability of each target class
        """
        total = float(len(data))
        result = len(class_feature_map[target]) / total
        return result

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
        N = exponent / denominator
        return N

    def get_prediction(self, test_vector):
        """
        :param test_vector: list of features to test on
        :return:
        For each x of the test_vector:
            1. Calculate the Normal Probability Density Function using N(x; µ, σ). eg P(sepal length | Iris-virginica)
            2. Multiply all Normal PDFs together to get the likelihood.
            3. Multiply the likelihood by the Prior Probability to calculate the Joint PDF. P(Iris-virginica)

        *** Joint PDF = P(sepal length | Iris-virginica) * P(Iris-virginica) ***

        Return the class with the largest Joint PDF
        """
        joint_pdf = {}
        for target, features in self.summaries.iteritems():
            total_features = len(features['summary'])
            likelihood = 0
            for index in range(total_features):
                mean = features['summary'][index]['mean']
                stdev = features['summary'][index]['stdev']
                x = test_vector[index]
                normal_pdf = self.normal_pdf(x, mean, stdev)
                likelihood = joint_pdf.get(target, 1) * normal_pdf
            prior_prob = features['prior_prob']
            joint_pdf[target] = prior_prob * likelihood
        best_target = max(joint_pdf, key=joint_pdf.get)
        return best_target

    def predict(self, test_set):
        """
        :param test_set: list of features to test on
        :return:
        Predict the likeliest target for each row of the test_set.
        Return a list of predicted targets.
        """
        predictions = []
        for row in test_set:
            result = self.get_prediction(row)
            predictions.append(result)
        return predictions

    def accuracy(self, test_set, predicted):
        """
        :param test_set: list of test_data
        :param predicted: list of predicted classes
        :return:
        Calculate the the avrage performance of the classifier.
        """
        correct = 0
        actual = [item[-1] for item in test_set]
        for x, y in zip(actual, predicted):
            if x == y:
                correct += 1
        return correct / float(len(test_set))


def main():
    """
    :return:
    Training and testing data
    """
    urls = {
        'iris': 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',  # ~ 94% accurracy
        'diabetes': 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',  # ~ 76% accuracy
        'redwine': 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',  # ~ 34% accuracy
        'adult': 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'  # ~ 65% accuracy. varries with sample sizes
    }
    weight = 0.67
    for title, url in urls.iteritems():
        nb = GaussNB()
        data = requests.get(url).content
        print '\n ************ \n'
        print 'Executing: %s dataset' % title
        if title in ['iris', 'diabetes']:
            data = nb.load_csv(data, clean=title, header=False, rows=False, delimeter=',')
        elif title in ['redwine']:
            data = nb.load_csv(data, clean=title, header=True, rows=False, delimeter=';')
        elif title in ['adult']:
            data = nb.load_csv(data, clean=title, header=True, rows=10000, delimeter=',')
        else:
            print 'Add title and url.'
            break
        train_list, test_list = nb.split_data(data, weight)
        print 'Split %s rows into train=%s and test=%s rows' % (len(data), len(train_list), len(test_list))
        nb.train(train_list, -1)
        predicted_list = nb.predict(test_list)
        accuracy = nb.accuracy(test_list, predicted_list)
        print 'Accuracy: %.3f' % accuracy

if __name__ == '__main__':
    main()

