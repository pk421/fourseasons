from src.data_retriever import read_redis
import numpy as np
import toolsx as tools
import math

from util.profile import profile

#@profile
def run_returns_analyzer():
	"""
	This is a simple script that will run a returns analysis for a list of stocks. You can specify a list of stocks to
	analyze and this will allow you to put the returns into a number of "bins." The number of bins is configurable. The
	result will be what could easily be the input into a chart chart shows the returns distributions. From this chart,
	a returns model could be built, that would show the mean and sigma of the returns.
	"""

	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/large_universe.csv'
	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	in_file.close()

	len_stock_list = len(stock_list)

	# stock_list = ['AAPL', 'GOOG', 'GLD', 'SLV', 'NEM', 'ABX', 'XOM', 'CVX']
	# stock_list = ['GLD']


	bins = create_bin_list(bin_step=60, bin_min=0.85, bin_max=1.15)
	len_bins = len(bins.keys())
	bins_keys = sorted(bins.keys())

	total_day_count = 0
	all_rets = np.empty(0)
	list_all_rets = []


	for number, stock in enumerate(stock_list):
		# stock = stock_list[0]
		stock_data = read_redis(stock=stock, db_number=0, to_disk=False)[0]

		simple_close_data = np.empty(len(stock_data))
		simple_date_data = []


		for k, v in enumerate(stock_data):
			simple_close_data[k] = v['Close']
			simple_date_data.append(v['Date'])

		len_data = len(simple_close_data)
		
		# this catches some issues with trying to retrieve indices, etc. that have no data
		if len_data == 0:
			continue

		total_day_count += len_data
		
		simple_returns_data = np.empty(len_data)
		print number, len_stock_list, len_data, stock, len(list_all_rets)#, np.mean(all_rets), np.std(all_rets)
		simple_returns_data[0] = 0	


		for x in xrange(1, len_data):
			reg_ret = simple_close_data[x] / simple_close_data[x-1]
			
			ret = math.log(simple_close_data[x] / simple_close_data[x-1])
			# all_rets = np.append(all_rets, ret)
			list_all_rets.append(ret)
			# simple_returns_data[x] = ret
			# print x, simple_date_data[x], simple_close_data[x], round(reg_ret, 6), round(ret, 4)

			for y in xrange(0, len_bins-1):
				l_bound = bins_keys[y]
				u_bound = bins_keys[y+1]

				# print ret, l_bound, u_bound
				if (reg_ret) <= u_bound:
					bins[u_bound] += 1
					break


	
	all_rets = np.append(all_rets, list_all_rets)
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/util/output.csv'
	out_file = open(location, 'w')
	for k in sorted(bins.keys()):
		print k, bins[k], np.mean(all_rets)
		out_file.write(str(k))
		out_file.write(',')
		out_file.write(str(bins[k]))
		out_file.write('\n')

	out_file.close()

	print "\nMean Return: ", np.mean(all_rets), "\nSigma: ", np.std(all_rets)
	print "\nTotal Days Analyzed: ", (total_day_count / 1000000.0), " Million days"
	return


def create_bin_list(bin_step=10, bin_min=0.95, bin_max=1.05):
	"""
	Helper function to return a list of all the bin parameters, thus making this aspect easily configurable.
	"""

	bin_diff = bin_max - bin_min
	bin_inc = bin_diff / bin_step

	bin_list = []
	bin_d = {}

	for x in xrange(0, bin_step + 1):
		bin_to_add = bin_min + (bin_inc * x)
		bin_list.append(bin_to_add)
		bin_d[bin_to_add] = 0
		# print bin_list

	return bin_d



	# sma = tools.simple_moving_average(simple_close_data, 50)

	# buy = np.zeros(len(stock_data))
	# delta = buy
	# delta_sma = np.empty(len(stock_data))
	# final_ret = delta_sma

	# buy[0:51] = 0
	# delta[0:51] = 0

	# flag = 0
	# for k, v in enumerate(sma):
	# 	final_ret[k] = 1

	# 	delta[k] = 0
	# 	if k >= 50:
	# 		# print k, sma[k], sma[k-1], sma[k-2]
			
	# 		delta[k] = (sma[k] - sma[k-1])
	# 		delta_sma[k] = np.mean(delta[k-4:k+1])
			

	# 	if delta_sma[k] > 0 and delta_sma[k-1] < 0:
	# 		# entry conditions for a trade, k is the day we enter	
	# 		flag = k
	# 		# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), round(delta_sma[k], 4), '\t\tBUY'
		
	# 	elif delta_sma[k] < 0 and delta_sma[k-1] > 0:
	# 		# exit conditions for the trade, k is the day we entered
	# 		if flag != 0:
	# 			ret = simple_close_data[k] / simple_close_data[flag]
	# 			final_ret[k] = ret
	# 			print ret, final_ret[k]	
	# 			# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), round(delta_sma[k], 4), round(ret, 4), '\t\tSELL\n'
	# 		flag = 0
		
	# 	else:
	# 		# print k, stock_data[k]['Date'], simple_close_data[k], sma[k], round(delta[k], 4), delta_sma[k]
	# 		pass

	# 	if final_ret[k] == 0:
	# 		print k, flag

	# total_return = np.product(final_ret)
	# print "Total Return: ", total_return

	# return


