import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as lines


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys, line_y):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
        line: List/array-like of co-ordinates of the line.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c=colour)
    line = lines.Line2D(xs, line_y, c='r')
    ax.add_line(line)
    plt.show()


def best_linear_line(xs, ys):

    # Set-up the matrix for x by adding a column of x
    matrix_x = np.c_[np.ones(len(xs)), xs]

    # Calculation of 𝐴=(𝑋^𝑇*𝑋)^−1*𝑋^𝑇*𝑌
    part1 = np.linalg.inv(np.matmul(matrix_x.T, matrix_x))
    part2 = np.matmul(part1, matrix_x.T)
    fit = np.matmul(part2, ys)

    # First column of fit is the y-intercept and gradient is second column
    bestfit = fit[0] + fit[1] * xs

    # Showing the graph with data points and line
    view_data_segments(xs, ys, bestfit)
    return bestfit


def error(ys, line):
    # Caculated using ∑𝑖(𝑦̂𝑖−𝑦𝑖)2  where 𝑦̂𝑖=𝑎+𝑏𝑥𝑖
    partsse = np.square(line - ys)
    sse = partsse.sum()
    return sse


test1 = load_points_from_file("train_data/basic_1.csv")
line_test1 = best_linear_line(test1[0], test1[1])
print(error(test1[1], line_test1))

