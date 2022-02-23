import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest as ztest
import csv


def getBaselineData():
    with open("experiments\\baseline.txt", newline="") as f:
        reader = csv.reader(f)
        baseline = list(reader)[0]
    return [float(i) for i in baseline if i != ""]


def getTournamentData():
    with open("experiments\\tournament_final.txt", newline="") as f:
        reader = csv.reader(f)
        tournament = list(reader)[0]
    return [float(i) for i in tournament if i != ""]


def twoSampleZTest():
    baseline = getBaselineData()
    tournament = getTournamentData()
    test_stat, pvalue = ztest(baseline, tournament, alternative="smaller")
    print("Two Sample Z-Test to check if the mean performance of the baseline is lower than Tournament Selection")
    print(f'Test Statistic: {test_stat}')
    print(f'p-vale: {pvalue}')
    return


if __name__ == "__main__":
    twoSampleZTest()