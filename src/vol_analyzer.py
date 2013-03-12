from src.data_retriever import read_redis
import numpy as np
import tools as tools

def run_vol_analyzer():

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
			flag = k
			# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), round(delta_sma[k], 4), '\t\tBUY'
		
		elif delta_sma[k] < 0 and delta_sma[k-1] > 0:
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
