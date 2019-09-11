import pandas as pd 
from datetime import datetime
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

models = ['Linear Regression', 'Logistic Regression', 'Bayesian Ridge']
mean_sq_err = []


df = pd.read_csv('INTC.csv')
df.set_index('Date')
new_data = df[['Date', 'Adj Close']].copy()

dates = [datetime.strptime(x, '%Y-%m-%d').toordinal() for x in list(new_data['Date'])]

prices = list(new_data['Adj Close'])

linear_mod = linear_model.LinearRegression()
	
dates = np.reshape(dates, (len(dates), 1))
prices = np.reshape(prices, (len(prices), 1))
linear_mod.fit(dates, prices)

	
def predict_price_linear(dates, prices, target_date):
	#Enter date as 'YYYY-mm-dd'

	linear_mod = linear_model.LinearRegression()
	target_date_convert = datetime.strptime(target_date, '%Y-%m-%d').toordinal()
	dates = np.reshape(dates, (len(dates), 1))
	prices = np.reshape(prices, (len(prices), 1))
	linear_mod.fit(dates, prices)

	predicted_prices = linear_mod.predict(dates)

	#predicted_price = linear_mod.predict(datetime.strptime(target_date, '%Y-%m-%d').toordinal())
	predicted_price = linear_mod.predict(np.reshape([target_date_convert], (1,-1)))
	
	#Find mean_squared_error
	error = mean_squared_error(prices, predicted_prices)
	print('Error: {:.4f}'.format(error))
	mean_sq_err.append(error)

	#visualize
	plt.scatter(dates, prices, color='yellow')
	plt.plot(dates, linear_mod.predict(dates), color='blue', linewidth=3)
	plt.show()

	return "Linear Regression: Stock Price for {0:} is ${1:.2f}".format(target_date, predicted_price[0][0])


def predict_price_logistic(dates, prices, target_date):
	#Enter date as 'YYYY-mm-dd'

	linear_mod = linear_model.LogisticRegression()
	target_date_convert = datetime.strptime(target_date, '%Y-%m-%d').toordinal()
	dates = np.reshape(dates, (len(dates), 1))
	prices = np.reshape(prices, (len(prices), 1))
	prices = [int(x) for x in prices]
	linear_mod.fit(dates, prices)

	predicted_prices = linear_mod.predict(dates)

	#predicted_price = linear_mod.predict(datetime.strptime(target_date, '%Y-%m-%d').toordinal())
	predicted_price = linear_mod.predict(np.reshape([target_date_convert], (1,-1)))

	#Find mean_squared_error
	error = mean_squared_error(prices, predicted_prices)
	print('Error: {:.4f}'.format(error))
	mean_sq_err.append(error)

	#visualize
	plt.scatter(dates, prices, color='yellow')
	plt.plot(dates, linear_mod.predict(dates), color='blue', linewidth=3)
	plt.show()

	#return predicted_price[0]
	return "Logistic Regression: Stock Price for {0:} is ${1:.2f}".format(target_date, predicted_price[0])


def predict_price_bayesian(dates, prices, target_date):
	#Enter date as 'YYYY-mm-dd'

	linear_mod = linear_model.BayesianRidge()
	target_date_convert = datetime.strptime(target_date, '%Y-%m-%d').toordinal()
	dates = np.reshape(dates, (len(dates), 1))
	prices = np.reshape(prices, (len(prices), 1))
	linear_mod.fit(dates, prices)

	predicted_prices = linear_mod.predict(dates)

	#predicted_price = linear_mod.predict(datetime.strptime(target_date, '%Y-%m-%d').toordinal())
	predicted_price = linear_mod.predict(np.reshape([target_date_convert], (1,-1)))

	#Find mean_squared_error
	error = mean_squared_error(prices, predicted_prices)
	print('Error: {:.4f}'.format(error))
	mean_sq_err.append(error)

	#visualize
	plt.scatter(dates, prices, color='yellow')
	plt.plot(dates, linear_mod.predict(dates), color='blue', linewidth=3)
	plt.show()

	return "Bayesian Regression: Stock Price for {0:} is ${1:.2f}".format(target_date, predicted_price[0])


def best_performing_model():
	smallest_err = min(mean_sq_err)
	index = mean_sq_err.index(smallest_err)

	return "{0:} gives the best prediction with a mean squared error of {1:.4f}".format(models[index], mean_sq_err[index])


print(predict_price_linear(dates, prices, '2018-10-10'))
print(predict_price_logistic(dates, prices, '2018-10-10'))
print(predict_price_bayesian(dates, prices, '2018-10-10'))

print('\n----------\n')
print(best_performing_model())
print('\n----------')
