from src.data_retriever import read_redis
import numpy as np
import toolsx as tools

def run_returns_analyzer():
	"""
	This is a simple script that will run a returns analysis for a list of stocks. You can specify a list of stocks to
	analyze and this will allow you to put the returns into a number of "bins." The number of bins is configurable. The
	result will be what could easily be the input into a chart chart shows the returns distributions. From this chart,
	a returns model could be built, that would show the mean and sigma of the returns.
	"""

	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/list_sp_500.csv'
	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	in_file.close()

	stock_list = ['AAPL', 'GOOG', 'GLD', 'SLV', 'NEM', 'ABX', 'XOM', 'CVX']
	stock_list = ['GLD']

	stock = stock_list[0]
	stock_data = read_redis(stock=stock, db_number=0, to_disk=False)[0]

	simple_close_data = np.empty(len(stock_data))

	for k, v in enumerate(stock_data):
		simple_close_data[k] = v['Close']

	sma = tools.simple_moving_average(simple_close_data, 50)

	buy = np.zeros(len(stock_data))
	delta = buy
	delta_sma = np.empty(len(stock_data))
	final_ret = delta_sma

	buy[0:51] = 0
	delta[0:51] = 0

	flag = 0
	for k, v in enumerate(sma):
		final_ret[k] = 1

		delta[k] = 0
		if k >= 50:
			# print k, sma[k], sma[k-1], sma[k-2]
			
			delta[k] = (sma[k] - sma[k-1])
			delta_sma[k] = np.mean(delta[k-4:k+1])
			

		if delta_sma[k] > 0 and delta_sma[k-1] < 0:
			# entry conditions for a trade, k is the day we enter	
			flag = k
			# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), round(delta_sma[k], 4), '\t\tBUY'
		
		elif delta_sma[k] < 0 and delta_sma[k-1] > 0:
			# exit conditions for the trade, k is the day we entered
			if flag != 0:
				ret = simple_close_data[k] / simple_close_data[flag]
				final_ret[k] = ret
				print ret, final_ret[k]	
				# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), round(delta_sma[k], 4), round(ret, 4), '\t\tSELL\n'
			flag = 0
		
		else:
			# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), delta_sma[k]
			pass

		if final_ret[k] == 0:
			print k, flag

	total_return = np.product(final_ret)
	print "Total Return: ", total_return

	return


