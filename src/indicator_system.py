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

	### in_file_name = 'tda_free_etfs'
	in_file_name = 'big_etfs'
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + in_file_name + '.csv'

	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	for k, item in enumerate(stock_list):
		new_val = item.split('\r')[0]
		stock_list[k] = new_val
	in_file.close()

	# stock_list = ['SPY', 'QQQ', 'DIA', 'TLT']
	paired_list = get_paired_stock_list(sorted(stock_list), fixed_stock='SPY')

	len_stocks = len(paired_list)

	interest_items = []
	trade_log = []

	current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
	out_file = open('/home/wilmott/Desktop/fourseasons/fourseasons/results/indicator_results_' + in_file_name + '_' + str(current_time) +'.csv', 'w')

	days_analyzed = 0
	for k, item in enumerate(paired_list):
		print k, len_stocks, item['stock_1'], item['stock_2']
		output, trades, x = do_indicator_test(item, k, len(paired_list))
		if x:
			days_analyzed += x
		if trades is not None and len(trades) > 0:
			trade_log.extend(trades)
	
	output_fields = ('stock_1', 'stock_2', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'entry_sma', \
					 'entry_rsi', 'exit_rsi', 'entry_mean_price', 'entry_sigma', 'entry_sigma_over_p', 'time_in_trade', \
					 'trade_result', 'ret', 'chained_ret')

	output_string = ','.join(output_fields)
	output_string += '\n'

	# The backtest_trade_log effectively shrinks the trade log into only those trades that would be 
	# possibly in a chronologically traded system (i.e. one at a time)
	trade_log = backtest_trade_log(trade_log)

	rets = []
	for trade_item in trade_log:
		rets.append(trade_item.chained_ret)

#		print trade_item.stock_2, trade_item.entry_date, trade_item.exit_date, trade_item.entry_price, trade_item.exit_price, trade_item.ret, trade_item.chained_ret, \
#				trade_item.entry_sigma, trade_item.entry_sigma_over_p
#		if trade_item.chained_ret < 0.0:
#			return

		output_string += ','.join([str(getattr(trade_item, item)) for item in output_fields])
		output_string += '\n'

	out_file.write(output_string)
	out_file.close()

	# Note that if there is a negative total_return, then the pow function will throw a domain error!!!!
	total_return = np.product(rets)
	geom_return = math.pow(total_return, (1.0/len(trade_log)))
	sharpe_ratio, total_days_in = get_sharpe_ratio(trade_log)

	total_years_in = total_days_in / 252
	annualized_return = math.pow(total_return, (1.0/total_years_in))


	print "\n\nTrades, Total, geom ret, ann ret", len(trade_log), total_return, geom_return, annualized_return
	print '\nDays Analyzed', days_analyzed

	print "\nFinished: ", len_stocks

def do_indicator_test(item, k, len_stocks):

	stock_1_data = manage_redis.parse_fast_data(item['stock_1'])
	stock_2_data = manage_redis.parse_fast_data(item['stock_2'])
	try:
		# print "Getting data for: ", item['stock_1'], item['stock_2']
		stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
	except:
		return None, None, None
	
	
	# rsi_stock_1 = tools.rsi(stock_1_close, 4)
	if len(stock_1_trimmed) < 201:
		return None, None, None

	days_analyzed = len(stock_1_trimmed) - 200

	rsi_stock_2 = tools.rsi(stock_2_close, 4)
	sma_stock_2 = tools.simple_moving_average(stock_2_close, 200)

	trade_log = []
	output = ''

	end_data = len(stock_1_close)
	output = None
	result = None
	next_index = 0

	for x in xrange(200, end_data):
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
		rsi_lower_bound = 25
		rsi_upper_bound = 100-rsi_lower_bound
		if p_0 > sma_0:
			if (rsi_1 > rsi_lower_bound and rsi_0 < rsi_lower_bound):
				result = trade_result()
				entry_signal = True

		else:
			if (rsi_1 < rsi_upper_bound and rsi_0 > rsi_upper_bound):
				result = trade_result()
				entry_signal = True

		# cancel entry if there is no volatility...
		if entry_signal:
			### Fixme: retest here...the indexing below is actually using 101 periods, not 100!!
			mu_price = np.mean(stock_2_close[x-100:x+1])
			sigma = np.std(stock_2_close[x-100:x+1])
			# if sigma / mu_price < 0.09:
#			if sigma / p_0 < 0.085:
			if sigma / p_0 < 0.095:
				entry_signal = False
				continue

		if entry_signal:
			result.stock_2 = item['stock_2']
			result.start_index = x
			result.entry_date = stock_2_trimmed[x]['Date']
			result.entry_price = p_0
			result.entry_rsi = rsi_0
			result.entry_sma = sma_0
			result.entry_mean_price = mu_price	# No longer really used
			result.entry_sigma = sigma
			result.entry_sigma_over_p = sigma / p_0

			if result.entry_rsi < rsi_lower_bound:
				result.long_short = 'long'
			elif result.entry_rsi > rsi_upper_bound:
				result.long_short = 'short'

			result, next_index = do_post_trade_analysis(stock_2_close, stock_2_trimmed, rsi_stock_2, sma_stock_2, \
								 x, result, rsi_lower_bound, rsi_upper_bound)

			if result:
				trade_log.append(result)


	return output, trade_log, days_analyzed


def do_post_trade_analysis(stock_2_close, stock_2_trimmed, rsi, sma, x, result, rsi_lower_bound, rsi_upper_bound):

	start_index = x+1
	len_data = len(stock_2_close)

	trading_up = True if result.long_short == 'long' else False
	trading_down = True if result.long_short == 'short' else False

	entry_sigma_over_p = result.entry_sigma_over_p
	# entry_sigma = entry_sigma_over_p * result.entry_mean_price
	entry_sigma = entry_sigma_over_p * result.entry_price 

	# this stop loss is in terms of the # of sigma
	stop_loss = 1.3
	pc_stop_loss = -0.5

	price_log = [stock_2_close[x]]

	for x in xrange(start_index, 999999):

		if x == len_data:
			return None, None

		date_today = stock_2_trimmed[x]['Date']
		current_price = stock_2_close[x]
		price_log.append(current_price)
		price_change_pc = (current_price - result.entry_price) / result.entry_price

		if trading_up:
			ret = price_change_pc
		else:
			ret = -price_change_pc

		# if trading_up and (rsi[x] > 55 or ret < -0.015):
		if trading_up and (rsi[x] > 55 or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
			ret <= pc_stop_loss):

			if ret > 0:
			   	# print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
			   	result.trade_result = "Profit"
				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.exit_rsi = rsi[x]
				result.end_index = x
				result.price_log = price_log
				return result, result.end_index

			else:
			   	# print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
			   	result.trade_result = "Loss"
				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.exit_rsi = rsi[x]
				result.end_index = x
				result.price_log = price_log
				return result, result.end_index

		# elif trading_down and (rsi[x] < 45 or ret < -0.015):
		elif trading_down and (rsi[x] < 45 or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
			ret <= pc_stop_loss):
			
			if ret > 0:
			   	# print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
			   	result.trade_result = "Profit"
				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.exit_rsi = rsi[x]
				result.end_index = x
				result.price_log = price_log
				return result, result.end_index

			else:
			   	#print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
			   	result.trade_result = "Loss"
				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.exit_rsi = rsi[x]
				result.end_index = x
				result.price_log = price_log
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
		self.long_short = ''

		self.entry_price = 0
		self.exit_price = 0
		self.entry_rsi = 0
		self.exit_rsi = 0
		self.entry_sma = 0
		self.entry_mean_price = 0
		self.entry_sigma = 0
		self.entry_sigma_over_p = 0

		self.price_log = []

		self.ret = 0
		self.chained_ret = 0
		self.time_in_trade = 0
		self.trade_result = '' # profit, loss, timeout



def backtest_trade_log(trade_log):
	print "Total Entries Found Length: ", len(trade_log)
	chrono_trade_log = trade_log
	chrono_trade_log.sort(key=lambda x: x.entry_date)

	# for result in chrono_trade_log:
		# print result.stock_2, result.entry_date, result.exit_date

	small_log = []
	len_data = len(chrono_trade_log)

	x = 0
	while x < len_data:

		if x == len_data - 1:
			small_log.append(chrono_trade_log[x])
			break

		current_date = chrono_trade_log[x].entry_date

		# We must skip trading opportunities if we have not exited the first trade!!
		if len(small_log) > 0:
			last_exit = small_log[-1].exit_date
			todays_entry = current_date

			last_exit = datetime.datetime.strptime(last_exit, '%Y-%m-%d')
			todays_entry = datetime.datetime.strptime(todays_entry, '%Y-%m-%d')

			# The continue here will cut out all other logic and move on to the next item in the
			# master trade log. Basically, if we are already in a trade and have not exited yet,
			# then nothing else matters, we must keep looking at other trades
			if todays_entry <= last_exit:
				x += 1
				continue

		# We are iterating over the entry date-sorted chrono trade log so we are guaranteed to 
		# only encounter trades in chrono order. The filter here finds ALL trades starting on the
		# day of the current trade, so that we can see all possibilities like in real life
		# Then append the best trade opportunity and incrememt the chrono trade log counter by the
		# number of trades that started today
		start_today = filter(lambda y: y.entry_date == current_date, chrono_trade_log)
		# print "start_today", current_date, len(start_today)
		
		start_today.sort(key=lambda z: z.entry_sigma_over_p, reverse=True)
		# print "here", [z.entry_sigma_over_p for z in start_today]
		small_log.append(start_today[0])
		
		x += len(start_today)

	return small_log

def get_sharpe_ratio(trade_log):

	# sharpe_ratio = 0

	# equity_list = [100]
	# start_day = trade_log[0].entry_date
	# start_day = datetime.datetime.strptime(exit_day, '%Y-%m-%d')
	
	# end_day = trade_log[-1].exit_date
	# exit_day = datetime.datetime.strptime(exit_day, '%Y-%m-%d')

	# for item in trade_log:

	# 	equity_list.append(equity_list[-1])

	### To properly determine the sharp ratio, we must compare the returns in the trade log with returns in
	# SPY over the same period of time. Specifically, we must know the number of days that SPY traded within the
	# time period of interest, then insert returns of zero in the trade log so that the length matches SPY

	ref_price_data = manage_redis.parse_fast_data('SPY')

	system_ret_log = []
	print "*****************************SHARPE RATIO ANALYSIS"
	for item in trade_log:
		# print item.entry_date, item.exit_date, item.entry_price, item.exit_price, item.ret, item.time_in_trade, len(item.price_log), item.price_log
		p = ('new_trade', item.long_short, item.price_log[0])
		system_ret_log.append(p)
		if len(item.price_log) > 1:
			for x in xrange(1, len(item.price_log)):
				p = ('existing_trade', item.long_short, item.price_log[x])
				system_ret_log.append(p)

	# print '\n\n\n', system_ret_log[-20:0], len(system_ret_log)

		
	first_entry = trade_log[0].entry_date
	last_exit = trade_log[-1].exit_date

	first_entry = datetime.datetime.strptime(first_entry, '%Y-%m-%d')
	last_exit = datetime.datetime.strptime(last_exit, '%Y-%m-%d')

	ref_trimmed_price_data = []
	for day in ref_price_data:
		z = datetime.datetime.strptime(day['Date'], '%Y-%m-%d')
		if z >= first_entry and z <= last_exit:
			# We hardcode "long" here because this is a buy and hold assumption...
			p = ('existing_trade', 'long', day['AdjClose'])
			ref_trimmed_price_data.append(p)


	

	mean, std, sharpe_ratio = get_returns(ref_trimmed_price_data)
	print "Reference Mu, Sigma, Sharpe, # Days: ", round(mean, 6), round(std, 6), round(sharpe_ratio, 6), len(ref_trimmed_price_data)

	mean, std, sharpe_ratio = get_returns(system_ret_log)
	print "\nSystem Mu, Sigma, Sharpe, #Days, Pct in Mkt", round(mean, 6), round(std, 6), round(sharpe_ratio, 6), len(system_ret_log), \
														 round(float(len(system_ret_log)) / len(ref_trimmed_price_data), 4)

	total_days = len(ref_trimmed_price_data)

	return sharpe_ratio, total_days


def get_returns(price_list):

	### FIXME: This is totally wrong right now. We should pass into this a list of tuples. The first item in the tuple
	# would indicate whether we should "reset" the reference point here to the previous value. That is, if the previous
	# price was 150, and the next price is 200, but is the start of a new trade, then we should be using 200 as the ref
	# point, rather than 150.

	# The input argument into this function is a 3-tuple of: ( new/existing trade , long/short, price )

	ret_list = []

	len_data = len(price_list)

	for k in xrange(0, len_data):
		if k == 0:
			continue

		if price_list[k][0] == 'new_trade':
			# we skip adding a "return" for the first day of a trade since we have not been in the trade overnight yet

			current_entry_price = price_list[k][2]
#			print k, price_list[k], (price_list[k][2] / price_list[k-1][2])
			continue

		# Strategy: Calculate baseline ret same safe way as always. If short, change sign as we did before. Then add
		# one to use chained return
		baseline_ret = (price_list[k][2] - price_list[k-1][2]) / price_list[k-1][2]
		if price_list[k][1] == 'short':
			baseline_ret = -baseline_ret
		current_ret = baseline_ret + 1


		ret_list.append(current_ret)
#		print k, price_list[k], current_ret

	mean = np.mean(ret_list) - 1
	std = np.std(ret_list)
	sharpe_ratio = mean / std

	return mean, std, sharpe_ratio

