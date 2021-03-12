import numpy as np


# function calculating gradient descent
def gradientDescent(x, y):
    a = 1
    b = 1
    epochs = 1000
    learning_rate = 0.01
    n = len(x)

    for epoch in range(epochs):
        yPredicted = a * x + b

        da = (1 / n) * sum(2 * (y - yPredicted) * -x)
        db = (1 / n) * sum(2 * (y - yPredicted) * -1)

        a = a - learning_rate * da
        b = b - learning_rate * db

        cost = (1 / n) * sum([val ** 2 for val in (y - yPredicted)])

        print("a {} b {} cost {} epoch {}".format(a, b, cost, epoch))


# main application
xData = np.array([1, 2, 3, 4])
yData = np.array([1, 3, 2, 3])

gradientDescent(xData, yData)
