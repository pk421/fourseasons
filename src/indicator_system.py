import numpy as np
import scipy as scipy
import datetime
import time

import math
import statsmodels.tsa.stattools as stats

import toolsx as tools

from util.memoize import memoize, MemoizeMutable

from data.redis import manage_redis
# from src import math_tools
from src.cointegrations_data import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly, \
									get_bunches_of_pairs


def run_indicator_system():


	sectors = ('basic_materials', 'conglomerates', 'consumer_goods', 'financial', 'healthcare', 'industrial_services', \
			   'services', 'technology', 'utilities')

	in_file_name = 'big_etfs'
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/sectors/' + in_file_name + '.csv'

	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	for k, item in enumerate(stock_list):
		new_val = item.split('\r')[0]
		stock_list[k] = new_val
	in_file.close()

	# stock_list = ['SPY']
	paired_list = get_paired_stock_list(sorted(stock_list), fixed_stock='SPY')

	len_stocks = len(paired_list)

	interest_items = []
	trade_log = []

	current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
	out_file = open('/home/wilmott/Desktop/fourseasons/fourseasons/indicator_results_' + in_file_name + '_' + str(current_time) +'.csv', 'w')
	# out_file_2 = open('/home/wilmott/Desktop/fourseasons/fourseasons/cointegration_results_synth_prices_' + str(current_time) + '.csv', 'w')

	days_analyzed = 0
	for k, item in enumerate(paired_list):
		print k, len_stocks, item['stock_1'], item['stock_2']
		output, trades, x = do_indicator_test(item, k, len(paired_list))
		if x:
			days_analyzed += x
		if trades is not None and len(trades) > 0:
			trade_log.extend(trades)
	
	output_fields = ('stock_1', 'stock_2', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'entry_rsi', \
					 'entry_sma', 'entry_mean_price', 'entry_sigma', 'entry_sigma_over_mu', 'time_in_trade', \
					 'trade_result', 'ret', 'chained_ret')

	output_string = ','.join(output_fields)
	output_string += '\n'

	rets = []
	for trade_item in trade_log:

		rets.append(trade_item.chained_ret)


		output_string += ','.join([str(getattr(trade_item, item)) for item in output_fields])
		output_string += '\n'

	out_file.write(output_string)
	out_file.close()

	total_return = np.product(rets)
	geom_return = math.pow(total_return, (1.0/len(trade_log)))

	print "\n\nTrades, Total, geom return", len(trade_log), total_return, geom_return

	print '\nDays Analyzed', days_analyzed

	print "\nFinished: ", len_stocks

def do_indicator_test(item, k, len_stocks):

	stock_1_data = manage_redis.parse_fast_data(item['stock_1'])
	stock_2_data = manage_redis.parse_fast_data(item['stock_2'])
	try:
		print "Getting data for: ", item['stock_1'], item['stock_2']
		stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
	except:
		return None, None, None
	
	
	# rsi_stock_1 = tools.rsi(stock_1_close, 4)
	if len(stock_1_trimmed) < 401:
		return None, None, None

	days_analyzed = len(stock_1_trimmed) - 400

	rsi_stock_2 = tools.rsi(stock_2_close, 4)
	sma_stock_2 = tools.simple_moving_average(stock_2_close, 200)

	trade_log = []
	output = ''

	end_data = len(stock_1_close)
	output = None
	result = None
	next_index = 0

	for x in xrange(400, end_data):
		# If we've been told we're still in a trade then we simply skip this day
		if x <= next_index:
			continue

		# end_index = x
		# stock_1_window = stock_1_close[x : end_index+1]
		# stock_2_window = stock_2_close[x : end_index+1]

		rsi_0 = rsi_stock_2[x]
		rsi_1 = rsi_stock_2[x-1]

		sma_0 = sma_stock_2[x]
		sma_1 = sma_stock_2[x-1]

		p_0 = stock_2_close[x]
		p_1 = stock_2_close[x-1]


		entry_signal = False
		if p_0 > sma_0:
			if (rsi_1 > 20 and rsi_0 < 20):
				result = trade_result()
				entry_signal = True

		else:
			if (rsi_1 < 80 and rsi_0 > 80):
				result = trade_result()
				entry_signal = True

		# cancel entry if there is no volatility...
		if entry_signal:
			mu_price = np.mean(stock_2_close[x-100:x+1])
			sigma = np.std(stock_2_close[x-100:x+1])
			if sigma / mu_price < 0.05:
				entry_signal = False
				continue

		if entry_signal:
			result.stock_2 = item['stock_2']
			result.start_index = x
			result.entry_date = stock_2_trimmed[x]['Date']
			result.entry_price = p_0
			result.entry_rsi = rsi_0
			result.entry_sma = sma_0
			result.entry_mean_price = mu_price
			result.entry_sigma = sigma
			result.entry_sigma_over_mu = sigma / mu_price

			result, next_index = do_post_trade_analysis(stock_2_close, stock_2_trimmed, rsi_stock_2, sma_stock_2, \
								 x, result)

			if result:
				trade_log.append(result)


	return output, trade_log, days_analyzed


def do_post_trade_analysis(stock_2_close, stock_2_trimmed, rsi, sma, x, result):

	start_index = x+1
	len_data = len(stock_2_close)

	trading_up = result.entry_rsi < 20
	trading_down = result.entry_rsi > 80

	for x in xrange(start_index, 999999):

		if x == len_data:
			return None, None

		date_today = stock_2_trimmed[x]['Date']
		current_price = stock_2_close[x]
		price_change_pc = (current_price - result.entry_price) / result.entry_price

		if trading_up:
			ret = price_change_pc
		else:
			ret = -price_change_pc

		if trading_up and (rsi[x] > 55 or ret < -0.015):

			if ret > 0:
			   	# print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
			   	result.trade_result = "Profit"
				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.end_index = x
				return result, result.end_index

			else:
			   	# print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
			   	result.trade_result = "Loss"
				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.end_index = x
				return result, result.end_index

		elif trading_down and (rsi[x] < 45 or ret < -0.015):
			
			if ret > 0:
			   	# print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
			   	result.trade_result = "Profit"
				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.end_index = x
				return result, result.end_index

			else:
			   	#print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
			   	result.trade_result = "Loss"
				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.end_index = x
				return result, result.end_index

	return result, result.end_index



class trade_result():

	def __init__(self):
		self.stock_1 = None
		self.stock_2 = None
		
		self.entry_date = None
		self.exit_date = None
		self.start_index = 0
		self.end_index = 0

		self.entry_price = 0
		self.exit_price = 0
		self.entry_rsi = 0
		self.entry_sma = 0
		self.entry_mean_price = 0
		self.entry_sigma = 0
		self.entry_sigma_over_mu = 0

		self.price_window = []

		self.ret = 0
		self.chained_ret = 0
		self.time_in_trade = 0
		self.trade_result = '' # profit, loss, timeout

