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
        line_y: List/array-like of y co-ordinates of the line.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20

    # Set-up the plot
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c=colour)

    # Splitting up and plotting lines to their segment
    list_x = np.split(xs, len_data / 20)
    for i in range(0, num_segments):
        line = lines.Line2D(list_x[i], line_y[:, i], c='r')
        ax.add_line(line)
    plt.show()


def best_linear_line(xs, ys):
    # Set-up the matrix for linear fit by adding a column of 1's
    matrix_x = np.c_[np.ones(len(xs)), xs]

    # Calculation of 𝐴=(𝑋^𝑇*𝑋)^−1*𝑋^𝑇*𝑌
    partfit1 = np.linalg.inv(np.matmul(matrix_x.T, matrix_x))
    partfit2 = np.matmul(partfit1, matrix_x.T)
    fit = np.matmul(partfit2, ys)

    # First column of fit is the y-intercept and gradient is second column
    bestfit = fit[0] + fit[1] * xs

    return bestfit


def best_poly_line(xs, ys):
    # Set-up matrix for poly fit (cube)
    matrix_x = np.c_[np.ones(len(xs)), xs, np.square(xs), np.power(xs, 3)]

    cube_part_fit1 = np.linalg.inv(np.matmul(matrix_x.T, matrix_x))
    cube_part_fit2 = np.matmul(cube_part_fit1, matrix_x.T)
    cube_fit = np.matmul(cube_part_fit2, ys)

    bestfit = cube_fit[0] + cube_fit[1] * xs + cube_fit[2] * np.square(xs) + cube_fit[3] * np.power(xs, 3)

    return bestfit


def best_sin_line(xs, ys):
    # Set-up the matrix for sin fit by adding a column of 1's and making a sin(x) column
    matrix_x = np.c_[np.ones(len(xs)), np.sin(xs)]

    # Calculation of 𝐴=(𝑋^𝑇*𝑋)^−1*𝑋^𝑇*𝑌
    partfit1 = np.linalg.inv(np.matmul(matrix_x.T, matrix_x))
    partfit2 = np.matmul(partfit1, matrix_x.T)
    fit = np.matmul(partfit2, ys)

    bestfit = fit[0] + fit[1] * np.sin(xs)

    return bestfit


def splitup(xs, ys):
    # Splitting the line into segments of 20 points
    list_x = np.split(xs, len(xs) / 20)
    list_y = np.split(ys, len(ys) / 20)
    num_segments = len(xs) // 20

    # Initialising matrix for best fit
    best_fit_linear = np.zeros([20, num_segments])
    best_fit_poly = np.zeros([20, num_segments])
    best_fit_sin = np.zeros([20, num_segments])

    best_fit = np.zeros([20, num_segments])
    min_error = 0

    # Calculate line for each segment
    for i in range(0, num_segments):
        # Calculating all different function types
        temp1 = best_linear_line(list_x[i], list_y[i])
        temp2 = best_poly_line(list_x[i], list_y[i])
        temp3 = best_sin_line(list_x[i], list_y[i])

        # Check and use the one with the lowest error
        best_fit_linear = np.hstack((best_fit_linear, np.reshape(temp1, (20, 1))))
        best_fit_linear = np.delete(best_fit_linear, 0, 1)

        best_fit_poly = np.hstack((best_fit_poly, np.reshape(temp2, (20, 1))))
        best_fit_poly = np.delete(best_fit_poly, 0, 1)

        best_fit_sin = np.hstack((best_fit_sin, np.reshape(temp3, (20, 1))))
        best_fit_sin = np.delete(best_fit_sin, 0, 1)

        error1 = error(list_y[i], best_fit_linear[:, num_segments - 1])
        error2 = error(list_y[i], best_fit_poly[:, num_segments - 1])
        error3 = error(list_y[i], best_fit_sin[:, num_segments - 1])

        best_temp_check = min([error1, error2, error3])
        if best_temp_check == error1:
            best_fit_temp = best_fit_linear[:, num_segments - 1]
            print("linear")
        elif best_temp_check == error2:
            best_fit_temp = best_fit_poly[:, num_segments - 1]
            print("cubic")
        elif best_temp_check == error3:
            best_fit_temp = best_fit_sin[:, num_segments - 1]
            print("sin")

        best_fit = np.hstack((best_fit, np.reshape(best_fit_temp, (20, 1))))
        best_fit = np.delete(best_fit, 0, 1)
        min_error = min_error + best_temp_check

    # Showing the graph with data points and line
    print(min_error)
    return best_fit


def error(ys, line):
    # Calculated using ∑𝑖(𝑦̂𝑖−𝑦𝑖)2  where 𝑦̂𝑖=𝑎+𝑏𝑥𝑖
    partsse = np.square(ys - line)
    sse = partsse.sum()
    return sse


def main(filename):
    test = load_points_from_file(filename)
    best_fit = splitup(test[0], test[1])
    if len(sys.argv) == 3:
        if sys.argv[2] == "--plot":
            view_data_segments(test[0], test[1], best_fit)


main(sys.argv[1])
