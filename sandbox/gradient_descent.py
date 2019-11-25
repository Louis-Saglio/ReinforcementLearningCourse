from random import random, randint

import numpy as np
from matplotlib import pyplot as plt
from numpy.ma import sin


def x_cube_plus_3_x_square_plus_7(x):
    return sin(x) ** 3 - 3 * sin(x) ** 2


def three_square_x_minus_6_x(x):
    return 3 * x ** 2 - 6 * x


def gradient_descent(function, derivative, x_new, x_prev, precision, learning_rate):
    x = np.linspace(-1, 3, 500)
    plt.plot(x, function(x))
    plt.show()

    x_list, y_list = [x_new], [function(x_new)]

    while abs(x_new - x_prev) > precision:
        x_prev = x_new
        x_new = x_prev - learning_rate * derivative(x_new)
        x_list.append(x_new)
        y_list.append(function(x_new))

    print("Local minimum occurs at: " + str(x_new))
    print("Number of steps: " + str(len(x_list)))

    plt.subplot(1, 2, 2)
    plt.scatter(x_list, y_list, c="g")
    plt.plot(x_list, y_list, c="g")
    plt.plot(x, function(x), c="r")
    plt.title("Gradient descent")

    plt.subplot(1, 2, 1)
    plt.scatter(x_list, y_list, c="g")
    plt.plot(x_list, y_list, c="g")
    plt.plot(x, function(x), c="r")
    plt.xlim([min(x_list), max(x_list)])
    plt.title("Zoomed in Gradient descent to Key Area")
    plt.show()


def main():
    gradient_descent(
        x_cube_plus_3_x_square_plus_7,
        three_square_x_minus_6_x,
        randint(-1, 2) + random(),
        0,
        0.001,
        0.05,
    )


if __name__ == "__main__":
    main()
