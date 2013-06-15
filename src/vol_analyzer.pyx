from src.data_retriever import read_redis
import numpy as np
import toolsx as tools

from util.profile import profile
from util.memoize import memoize


# from libc.math cimport log as log

# cpdef run_vol_analyzer():
	
# 	cdef float x, y, z

# 	for x in xrange(3, 10000000):
# 		y = log(x)
# 		z = log(y)
# 		if x % 1000000 == 0:
# 			print x, y, z

@memoize
def get_data():
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/list_sp_500.csv'
	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	in_file.close()

	stock_list = ['AAPL', 'GOOG', 'GLD', 'SLV', 'NEM', 'ABX', 'XOM', 'CVX']
	stock_list = ['SPY']

	stock = stock_list[0]
	stock_data = read_redis(stock=stock, db_number=0, to_disk=False)[0]
	return stock_data

@profile
def run_vol_analyzer():

	"""
	This is not really a volatility analyzer. It actually does a simple entry/exit analysis of trades based on the
	changes in short term SMAs. The algo started by attempting to look into inflection points in the SMA (second
	derivative). Now, however, it is simply looking at changes in the slope of the SMA (first derivative). It then
	applies a simple smoothing to this factor (see "delta_sma") to reduce noise. I think the second derivative is worth
	reinvestigating.
	"""

	stock_data = get_data()
	simple_close_data = np.empty(len(stock_data))

	for k, v in enumerate(stock_data):
		simple_close_data[k] = v['Close']

	sma = tools.simple_moving_average(simple_close_data, 20)

	delta = np.zeros(len(stock_data))
	delta_sma = np.empty(len(stock_data))
	final_ret = np.empty(len(stock_data))

	delta[0:51] = 0

	trade_entry_index = 0
	for k, v in enumerate(sma):
		final_ret[k] = 1.0

		if k >= 50:
			# print k, sma[k], sma[k-1], sma[k-2]		
			# delta is simply the one day change in the sma
			delta[k] = (sma[k] - sma[k-1])	
			delta_sma[k] = np.mean(delta[(k-4):(k+1)])			

		# Trade entry
		if delta_sma[k] > 0 and delta_sma[k-1] < 0:
			# entry conditions for a trade, k is the day we enter	
			if trade_entry_index == 0:
				trade_entry_index = k
			# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), round(delta_sma[k], 4), '\t\tBUY'
		
		# Trade exit
		elif delta_sma[k] < 0 and delta_sma[k-1] > 0:
			# exit conditions for the trade, k is the day we exit
			if trade_entry_index != 0:
				ret = simple_close_data[k] / simple_close_data[trade_entry_index]
				final_ret[k] = ret
				# print "Ret zero in exit: ", ret, final_ret[k], k, trade_entry_index
				# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), round(delta_sma[k], 4), round(ret, 4), '\t\tSELL\n'
			trade_entry_index = 0

		else:
			# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), delta_sma[k]
			pass

		if final_ret[k] == 0:
			print k, trade_entry_index, final_ret[k], simple_close_data[k], simple_close_data[trade_entry_index]

	for k, v in enumerate(final_ret):
		if v != 1.0:
			# print k, stock_data[k]['Date'], v
			pass

	total_return = np.prod(final_ret)
	# print "Total Return: ", total_return

	return


