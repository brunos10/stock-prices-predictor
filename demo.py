import csv
import numpy as np
import sklearn.svm as SVR
import matplotlib.pyplot as plt
import os

# tutorial: https://www.youtube.com/watch?v=SSu00IRRraY

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csv_file:
        csv_file_reader = csv.reader(csv_file)
        next(csv_file_reader)
        for row in csv_file_reader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, len(dates), 1)

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict_prices(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict_prices(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict_prices(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Suport Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


get_data('aapl.csv')

predicted_prices = predict_prices(dates, prices, 29)

print(predicted_prices)
