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
            target_map[x].append(features[:-1])
        return dict(target_map)

    def mean(self, numbers):
        """
        :param numbers: list of numbers
        :return:
        """
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

    def summarize(self, data):
        """
        :param data: lists of events (rows) in a list
        :return:
        Use zip to line up each feature into a single column across multiple lists.
        yield the mean and the stdev for each feature column.
        """
        for feature_col in zip(*data):
            yield {
                'stdev': self.stdev(feature_col),
                'mean': self.mean(feature_col)
            }

    def prior_prob(self, group, target, data):
        """
        :return:
        The probability of each target class
        """
        total = float(len(data))
        result = len(group[target]) / total
        return result

    def train(self, train_list, target):
        """
        :param data: train data
        :param target: target class
        :return:
        For each target:
            1. yield prior_prob: the probability of each class. P(class) eg P(Iris-virginica)
            2. yield summary: list of {'mean': 0.0, 'stdev': 0.0}
        """
        group = self.group_by_class(train_list, target)
        self.summaries = {}
        for target, features in group.iteritems():
            self.summaries[target] = {
                'prior_prob': self.prior_prob(group, target, train_list),
                'summary': [i for i in self.summarize(features)],
            }
        return self.summaries

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
        normal_prob = exponent / denominator
        return normal_prob

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
            prior_prob = features['prior_prob']
            for index in range(len(pdfs)):
                normal_prob = pdfs[index]
                predictors.append(prior_prob * normal_prob)
        marginal_prob = sum(predictors)
        return marginal_prob

    def posterior_probabilities(self, test_row):
        """
        :param test_row: single list of features to test
        :return:
        For each feature (x) in the test_row:
            1. Calculate Predictor Prior Probability using the Normal PDF N(x; µ, σ). eg = P(feature | class)
            2. Calculate Likelihood by getting the product of the prior and the Normal PDFs
            3. Multiply Likelihood by the prior to calculate the Joint PDF. P(Iris-virginica)

        E.g.
        prior_prob: P(setosa)
        likelihood: P(sepal length | setosa) * P(sepal width | setosa) * P(petal length | setosa) * P(petal width | setosa)
        numerator (joint pdf): prior_prob * likelihood
        denominator (marginal pdf): predictor prior probability
        posterior_prob = joint pdf/ marginal pdf

        returning a dictionary mapping of class to it's posterior probability
        """
        posterior_probs = {}
        for target, features in self.summaries.iteritems():
            total_features = len(features['summary'])
            likelihood = 0
            normal_probs = []
            for index in range(total_features):
                mean = features['summary'][index]['mean']
                stdev = features['summary'][index]['stdev']
                test_feature = test_row[index]
                normal_prob = self.normal_pdf(test_feature, mean, stdev)
                likelihood = posterior_probs.get(target, 1) * normal_prob
                normal_probs.append(normal_prob)
            marginal = self.marginal_pdf(normal_probs)
            prior_prob = features['prior_prob']
            posterior_probs[target] = (prior_prob * likelihood) / marginal
        return posterior_probs

    def get_prediction(self, test_row):
        """
        :param test_row: single list of features to test
        :return:
        Return the target class with the largest/best posterior probability
        """
        posterior_probs = self.posterior_probabilities(test_row)
        best_target = max(posterior_probs, key=posterior_probs.get)
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
        Calculate the the average performance of the classifier.
        """
        correct = 0
        actual = [item[-1] for item in test_set]
        for x, y in zip(actual, predicted):
            if x == y:
                correct += 1
        return correct / float(len(test_set))

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
    predicted = nb.predict(test_list)
    accuracy = nb.accuracy(test_list, predicted)
    print 'Accuracy: %.3f' % accuracy

if __name__ == '__main__':
    main()