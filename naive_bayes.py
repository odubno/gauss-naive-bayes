# -*- coding: utf-8 -*-
import random
import csv
from collections import defaultdict
from math import e
from math import pi
import requests


class GaussNB:

    def __init__(self):
        """
        https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
        """
        self.summaries = {}

    def load_csv(self, data):
        """
        :param data:
        :return:
        Load and covert each string into float
        """
        lines = csv.reader(data.splitlines())
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        return dataset

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
        Calculates the standard deviation for a list of numbers.
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
        :param data: training set
        :param target: identify class value column
        :return:
        Creating a map of target to a list of it's features.
        One to many relationship.
        """
        target_map = defaultdict(list)
        for index in range(len(data)):
            features = data[index]
            x = features[target]
            target_map[x].append(features[:-1])
        return dict(target_map)

    def summarize(self, data):
        """
        :param data:
        :return:
        Use zip to line up each index of multiple lists into a single column.
        Calculate mean and stdev for each column of the target set
        """
        for attributes in zip(*data):
            summary = (self.mean(attributes), self.stdev(attributes))
            yield summary

    def train(self, data, x_target):
        """
        :param data:
        :param x_target: dependent variable
        :return:
        Return mean and stdev for each target target.
        """
        class_feature_map = self.group_by_target(data, x_target)
        self.summaries = {}
        for target, features in class_feature_map.iteritems():
            # summaries (mean, stdev) for each column
            self.summaries[target] = [i for i in self.summarize(features)]
        return self.summaries

    def normal_pdf(self, x, mean, stdev):
        """
        :param x: a variable
        :param mean: µ - the expected value or average from M samples
        :param stdev: σ - standard deviation
        Gaussian (Normal) Density function.
        N(x; µ, σ) = (1 / 2πσ) * (e ^ (x–µ)^2/-2σ^2
        :return:
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
        :param test_vector: list of features to test
        :return:
        Return the best predicted target for each row of the test_set.
        For each index of the test_vector, calculate the normal PDF
        Use Naive Bayes, N(x; µ, σ), to determine probabilities and multiply probabilities across each target.
        Return the target with the largest probability
        """
        probs = {}
        for target, features in self.summaries.iteritems():
            for index in range(len(features)):
                mean, stdev = features[index]
                x = test_vector[index]
                N = self.normal_pdf(x, mean, stdev)
                prob = probs.get(target, 1) * N
                probs[target] = prob
        best_target = max(probs, key=probs.get)
        return best_target

    def predict(self, test_set):
        """
        :param test_set: list of lists containing data points for each column.
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
        correct = 0
        actual = [item[-1] for item in test_set]
        for x, y in zip(actual, predicted):
            if x == y:
                correct += 1
        return correct / float(len(test_set))

if __name__ == '__main__':
    nb = GaussNB()
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
    data = requests.get(url).content
    # data = open('pima-indians-diabetes.data.csv').read()
    weight = 0.67
    data = nb.load_csv(data)
    train_list, test_list = nb.split_data(data, weight)
    print ('Split {0} rows into train={1} and test={2} rows').format(len(data), len(train_list), len(test_list))
    nb.train(train_list, -1)
    predicted_list = nb.predict(test_list)
    accuracy = nb.accuracy(test_list, predicted_list)
    print ('Accuracy: {0}%').format(accuracy)

