import numpy as np
import scipy as scipy
import datetime
import time

from math import log
import statsmodels.tsa.stattools as stats

import toolsx as tools

from util.memoize import memoize, MemoizeMutable

from data.redis import manage_redis
# from src import math_tools
from src.cointegrations_data import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly, \
									get_bunches_of_pairs


def run_cointegrations():
	"""
	Takes in data from stocks, and performs a basic ADF test on the data.
	"""

	sectors = ('basic_materials', 'conglomerates', 'consumer_goods', 'financial', 'healthcare', 'industrial_services', \
			   'services', 'technology', 'utilities')

	## location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/300B_1M_and_etfs_etns.csv'
	

	in_file_name = 'basic_materials'
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/sectors/' + in_file_name + '.csv'

	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	for k, item in enumerate(stock_list):
		new_val = item.split('\r')[0]
		stock_list[k] = new_val
	in_file.close()

	# stock_list = ['ESV', 'RDC']

	paired_list = get_paired_stock_list(sorted(stock_list), fixed_stock=None)
	# paired_list = get_bunches_of_pairs()

	len_pairs = len(paired_list)

	interest_items = []
	trade_log = []

	current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
	out_file = open('/home/wilmott/Desktop/fourseasons/fourseasons/cointegration_results_' + in_file_name + '_' + str(current_time) +'.csv', 'w')
	# out_file_2 = open('/home/wilmott/Desktop/fourseasons/fourseasons/cointegration_results_synth_prices_' + str(current_time) + '.csv', 'w')

	for k, item in enumerate(paired_list):
		print k, len_pairs, item['stock_1'], item['stock_2']
		output, trades = do_cointegration_test(item, k, len(paired_list))
		if trades is not None and len(trades) > 0:
			trade_log.extend(trades)

	############
	# if len(paired_list) == 1:
	# 	out_file_2.write(str(output))
	# 	out_file_2.close()
	############

	output_fields = ('stock_1', 'stock_2', 'entry_date', 'window_length', 'price_correlation_coeff', \
					 'df_test_statistic', 'df_pc_crit_val', 'df_p_value', 'df_decay_half_life', 'synthetic_mu', \
					 'synthetic_sigma', 'sigma_price_pc', 'rsi_most_extreme', 'entry_min_extreme_threshold_sigma', \
					 'entry_actual_entry_sigma', 'trade_result', 'time_in_trade', 'synth_price_start', \
					 'synth_price_end', 'ret', 'synth_chained_ret', )
	output_string = ','.join(output_fields)
	output_string += '\n'

	for trade_item in trade_log:

		output_string += ','.join([str(getattr(trade_item, item)) for item in output_fields])
		output_string += '\n'

	out_file.write(output_string)
	out_file.close()


	print "\nFinished: ", len_pairs


def do_cointegration_test(item, k, len_paired_list):

	stock_1_data = manage_redis.parse_fast_data(item['stock_1'])
	stock_2_data = manage_redis.parse_fast_data(item['stock_2'])
	try:
		print "Getting data for: ", item['stock_1'], item['stock_2']
		stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
	except:
		return None, None

	rsi_stock_1 = tools.rsi(stock_1_close, 5)
	rsi_stock_2 = tools.rsi(stock_2_close, 5)

	trade_log = []

	end_data = len(stock_1_close)
	output = None
	result = None

	window_size = 150
	next_index = 0
	# We go to twice the window size to allow one window size of time for post trade analysis
	for x in xrange(0, end_data - (2 * window_size)):
		# If we've been told we're still in a trade then we simply skip this day
		if x <= next_index:
			continue

		# print "after test", x, next_index, end_data - (2 * window_size)

		end_index = x + window_size
			
		stock_1_window = stock_1_close[x : end_index+1]
		stock_2_window = stock_2_close[x : end_index+1]

		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stock_1_window, stock_2_window)
		lin_reg_results = (slope, intercept, r_value, p_value, std_err)

		# Here, we will only consider stocks where we can first see that a linreg of their prices can be done with
		# a high r-value. If we can't even find a good correlation between the stocks' prices, we skip. This is like
		# a sanity check - we should not be trying to trade 'MSFT' and 'XOM' as a pair trade!!!
		
		# print "Start: ", item['stock_1'], stock_1_trimmed[x]['Date'], stock_1_trimmed[x]['Close'], stock_1_close[x], rsi_stock_1[x]
		one_rsi_at_extreme = rsi_stock_1[x] < 30 or rsi_stock_1[x] > 70 or rsi_stock_2[x] < 30 or rsi_stock_2[x] > 70
		rsi_most_extreme = max(abs(rsi_stock_1[x] - 50), abs(rsi_stock_2[x] - 50))
		# one_rsi_at_extreme = True
		
		if r_value >= 0.90 and one_rsi_at_extreme:
			## print "Start: ", item['stock_1'], item['stock_2'], stock_1_trimmed[x]['Date'], stock_1_close[x], rsi_stock_1[x]
			output, result, next_index = create_synthetic_investment(stock_1_close, stock_2_close, x, end_index, stock_1_trimmed, \
												 stock_2_trimmed, lin_reg_results, k, len_paired_list, rsi_most_extreme)


			if result:
				trade_log.append(result)
			# print "Window of high correlation used for prev. ADF Test: ", x, end_index
			if output is not None:
				# print "\nFound a good window: "
				# print "Start: ", stock_1_trimmed[x]['Date'], stock_1_close[x], rsi_stock_1[x]
				# print "End: ", stock_1_trimmed[end_index]['Date'], stock_1_close[x], rsi_stock_1[x]
				# out_file.write(output)
				# out_file.close
				# return
				# return output, trade_log
				pass
		else:
			# print "PRICES NOT CORRELATED", x
			next_index = 0
			pass

	return output, trade_log



def create_synthetic_investment(stock_1_close, stock_2_close, start_index, end_index, stock_1_trimmed, \
							    stock_2_trimmed, lin_reg_results, k, len_paired_list, rsi_most_extreme):
	"""
	This uses the hedge ratio (slope) of the regression to create a synthetic investment and create a list of all the
	residuals of that sythetic investment based on the hedge ratio. If the hedge ratio were perfect, the synthetic
	investment would always have a value of zero, since differencing is done. However, to trade it we look for a
	synthetic investment that has residual values that fluctuate around zero with some level of predictability. We use
	the Dickey Fuller test to check stationarity of the synthetic investment.

	Performs an ADF test using a particular window of data for the stocks, given that we already know the prices are
	highly correlated within the window of interest. The "trimmed" data here makes it much easier to print out the
	dates of the data for debugging purposes.
	"""
			
	stock_1_window = stock_1_close[start_index : end_index]
	stock_2_window = stock_2_close[start_index : end_index]
	len_stock_1_window = len(stock_1_window)

	slope, intercept, r_value, p_value, std_err = lin_reg_results

	### Remember for the regressions: if we have this: first = [0,1,2,3,4] and second = [0,2,4,6,8] and we do:
	# linregress(first, second), we get an equation of second = slope*first + intercept
	
	synthetic_price = 0
	synthetic_prices = []
	synthetic_price_pc = 0
	synthetic_prices_pc = []
	
	output_string = ''

	hedge_ratio = slope

	# print "Start: ", stock_1_close[start_index], stock_2_close[start_index], price_ratio_1_2, hedge_ratio
	# print "\n\n"
	
	stock_1_mean_price = np.mean(stock_1_window)
	stock_2_mean_price = np.mean(stock_2_window)
	mean_total_value = (hedge_ratio * stock_1_mean_price) + stock_2_mean_price

	previous_position_value = 0
	for x in xrange(start_index, end_index):

		# adding intercept simply corrects for an offset that would otherwise exist when the data is output
		synthetic_price = (hedge_ratio * stock_1_close[x]) - stock_2_close[x] + intercept
		synthetic_prices.append(synthetic_price)

		# synthetic_price_pc = synthetic_price / ((hedge_ratio * stock_1_close[start_index]) + stock_2_close[start_index] + intercept)
		# synthetic_price_pc = synthetic_price / ((hedge_ratio * stock_1_close[x]) + stock_2_close[x] + intercept)
		synthetic_price_pc = synthetic_price / mean_total_value
		synthetic_prices_pc.append(synthetic_price_pc)

		date = stock_1_trimmed[x]['Date']
		### print x, date, slope, intercept, stock_1_close[x], stock_2_close[x], current_residual, total_residual

		output_string += ','.join([
								   # str(spread),
								   str(stock_1_close[x]), 
								   str(stock_2_close[x]),
								   str(0),
								   str(synthetic_price / mean_total_value),
								   str(0),
								   # str(smoothed_total_residual)
								   ]) + '\n'
		## print x, date, round(total_residual, 4), round(total_residual_pc, 6), '%'


	rslope, rintercept, rr_value, rp_value, rstd_err = scipy.stats.linregress(range(len_stock_1_window), \
																					synthetic_prices)
	
	# print '\nTotal residuals slope, int, p, r: %.4f, %.4f, %.4f, %.4f' % (rslope, rintercept, rp_value, rr_value)
	# print '\nStock prices r, slope, intercept: %.4f, %.4f, %.4f' % (r_value, slope, intercept)
	
	mean_synth_price = round(np.mean(synthetic_prices), 5)
	sigma_synth_price = round(np.std(synthetic_prices), 5)
	
	sigma_price_pc = sigma_synth_price / mean_total_value
	if sigma_price_pc < 0.01:
		# print "SIGMA CUT: %.4f %.4f %.4f" % (sigma_synth_price, mean_total_value, sigma_price_pc)
		return None, None, 0

	# We first run a sanity check before even attempting a DF test - here, we verify that the r value of the synthetic
	# investment is somewhat low - it should be, indicating that it is not particularly correlated to the time. 
	##if abs(rr_value) <= 0.5:
		# print 'Synth inv: %i %.4f %.4f %.4f %.4f %.4f' % (start_index, rslope, rintercept, rr_value, rp_value, rstd_err)
	df_result = do_df_test(synthetic_prices)
	# else:
		# print 'SKIPPED: ', start_index
		# return None, None, 0

	if df_result and df_result[0] <= df_result[4]['10%'] and df_result[1] < 0.1:
		# print '%i %i %i %s %s %s %.3f %.3f %.4f %.3f %.4f' % (k, len_paired_list, start_index, \
		# 							stock_1_trimmed[0]['Symbol'], stock_2_trimmed[0]['Symbol'], \
		# 							stock_1_trimmed[start_index]['Date'], df_result[0], \
		# 							df_result[4]['5%'], df_result[1], slope, intercept)
		

		today = synthetic_prices[len_stock_1_window - 1]
		yesterday = synthetic_prices[len_stock_1_window - 2]
		lower_bound = (-1.5 * sigma_synth_price) + mean_synth_price
		upper_bound = (1.5 * sigma_synth_price) + mean_synth_price

		second_lower_bound = (-1.5 * sigma_synth_price / 3) + mean_synth_price
		second_upper_bound = (1.5 * sigma_synth_price / 3) + mean_synth_price

		# if (today > lower_bound and yesterday < lower_bound) or (today < upper_bound and yesterday > upper_bound):
		# print "Upper bound, today, yesterday: %.4f %.4f %.4f" % (upper_bound, today, yesterday)
		## if (yesterday < lower_bound and yesterday < today) or (yesterday > upper_bound and yesterday > today):
		entry_min_extreme_threshold_sigma = 0
		if ((yesterday < lower_bound and today > lower_bound) or (yesterday > upper_bound and today < upper_bound)) and \
			((today > second_upper_bound) or (today < second_lower_bound)):
		# if today < lower_bound or today > upper_bound:

			#print "Synth investment stats: %.4f %.4f %.4f %.4f %.4f" % (mean_synth_price, sigma_price_pc, \
			#										upper_bound, today, yesterday)
			result = trade_result()

			result.stock_1 = stock_1_trimmed[0]['Symbol']
			result.stock_2 = stock_2_trimmed[0]['Symbol']
			result.window_length = end_index - start_index
			result.entry_date = stock_1_trimmed[start_index]['Date']
			result.slope = slope
			result.intercept = intercept

			result.price_correlation_coeff = r_value
			result.df_test_statistic =df_result[0]
			result.df_pc_crit_val = df_result[4]['10%']
			result.df_p_value = df_result[1]
			result.df_decay_half_life = -(log(2,10) / result.df_test_statistic) * result.window_length
			
			result.synthetic_mu = mean_synth_price
			result.synthetic_sigma = sigma_synth_price
			
			result.mean_total_value = mean_total_value
			result.sigma_price_pc = sigma_synth_price / mean_total_value

			result.synthetic_prices = synthetic_prices

			result.rsi_most_extreme = rsi_most_extreme
			result.entry_min_extreme_threshold_sigma = entry_min_extreme_threshold_sigma
			result.entry_actual_entry_sigma = today / upper_bound

			result = do_post_trade_analysis(stock_1_close, stock_2_close, start_index, end_index, result)

			return output_string, result, result.end_index

	elif df_result:
		# print "FAILED DF TEST: ", start_index, result[0], result[4]['5%'], result[1], round(slope, 4), round(intercept, 4)
		pass
	
	return None, None, 0

def do_df_test(prices):
	# see here: http://statsmodels.sourceforge.net/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller
	# result is in the format: test statistic (lambda), p_value, used lag, # of observations, critical values
	result = stats.adfuller(prices)
	return result

def do_post_trade_analysis(stock_1_close, stock_2_close, start_index, end_index, result):

	len_window = result.window_length
	start_index = end_index # End index here represents the first day *after* the entry date of the extreme event

	slope = result.slope
	intercept = result.intercept
	synth_sigma = result.synthetic_sigma
	synth_sigma_loss = synth_sigma

	starting_synth_price = result.synthetic_prices[-1]
	entry_total_value = (slope * stock_1_close[start_index - 1]) + stock_2_close[start_index - 1]

	trading_up = starting_synth_price < 0
	trading_down = starting_synth_price >= 0

	# stop_loss = abs(starting_synth_price)

	in_trade_prices = []

	rets = []

	trading_up_stop_loss = (starting_synth_price - (1.6 * synth_sigma_loss))
	trading_down_stop_loss = (starting_synth_price + (1.6 * synth_sigma_loss))
	
	trading_up_profit_target = (result.synthetic_mu - (0.5 * synth_sigma))
	trading_down_profit_target = (result.synthetic_mu + (0.5 * synth_sigma))


	for x in xrange(start_index, start_index + result.window_length):

		current_price = (slope * stock_1_close[x]) - stock_2_close[x] + intercept
		price_change_pc = (current_price - starting_synth_price) / entry_total_value

		# print "Current Price: ", start_index, start_index + result.window_length, x, current_price, price_change_pc

		if trading_up:
			ret = price_change_pc
		else:
			ret = -price_change_pc
		rets.append(ret)

		# if current_price > result.synthetic_mu and starting_synth_price < result.synthetic_mu or \
		#    current_price < result.synthetic_mu and starting_synth_price > result.synthetic_mu:

		if trading_up and current_price > trading_up_profit_target or \
		   trading_down and current_price < trading_down_profit_target:

		   	print "Profit: ", starting_synth_price, current_price, ret, '\n'
		   	result.trade_result = "Profit"
			result.time_in_trade = x - (start_index - 1)
			result.synth_price_start = starting_synth_price
			result.synth_price_end = current_price
			result.ret = ret
			result.synth_chained_ret = 1 + ret
			result.end_index = x
			return result


		if trading_up and current_price < trading_up_stop_loss:
			print "Loss: ", starting_synth_price, current_price, ret, '\n'
			result.trade_result = "Loss"
			result.time_in_trade = x - (start_index - 1)
			result.synth_price_start = starting_synth_price
			result.synth_price_end = current_price
			result.ret = ret
			result.synth_chained_ret = 1 + ret
			result.end_index = x
			return result


		elif trading_down and current_price > trading_down_stop_loss:
			print "Loss: ", starting_synth_price, current_price, ret, '\n'
			result.trade_result = "Loss"
			result.time_in_trade = x - (start_index - 1)
			result.synth_price_start = starting_synth_price
			result.synth_price_end = current_price
			result.ret = ret
			result.synth_chained_ret = 1 + ret
			result.end_index = x
			return result

		# if x - (start_index - 1) > (0.1 * len_window):
		if x - (start_index - 1) > 25:
			print "Timeout: ", starting_synth_price, current_price, ret, '\n'
			result.trade_result = "Timeout"
			result.time_in_trade = x - (start_index - 1)
			result.synth_price_start = starting_synth_price
			result.synth_price_end = current_price
			result.ret = ret
			result.synth_chained_ret = 1 + ret
			result.end_index = x
			return result

		#if (x - start_index) == 3 or (x - start_index) == 6 or (x - start_index) == 9:
		# if (x - start_index) == 10:
			# synth_sigma_loss = synth_sigma_loss * (0.75)
			# synth_sigma = synth_sigma * 2

		# if (x - start_index) == 15:
		# 	synth_sigma = synth_sigma * 1.5

		if trading_up and current_price < (starting_synth_price - (1.0 * synth_sigma_loss)) or \
		   trading_down and current_price > (starting_synth_price + (1.0 * synth_sigma_loss)):
			trading_up_profit_target = starting_synth_price
			trading_down_profit_target = starting_synth_price

		# if x - (start_index - 1) == 10:
		# 	trading_up_profit_target = starting_synth_price
		# 	trading_down_profit_target = starting_synth_price

	return result


class trade_result():

	def __init__(self):
		self.stock_1 = None
		self.stock_2 = None
		self.start_date = None
		self.slope = 0
		self.intercept = 0
		
		self.end_index = 0

		self.window_length = 0
		self.price_correlation_coeff = 0
		
		self.df_test_statistic = 0
		self.df_pc_crit_val = 0
		self.df_p_value = 0
		self.df_decay_half_life = 0

		self.synthetic_mu = 0
		self.synthetic_sigma = 0
		
		self.mean_total_value = 0
		self.sigma_price_pc = 0

		self.synthetic_prices = []

		self.synth_price_start = 0
		self.synth_price_end = 0
		self.synth_return = 0
		self.synth_chained_ret = 0
		self.synth_time_in_trade = 0
		self.trade_result = '' # profit, loss, timeout

		###
		# Input parameters
		self.rsi_most_extreme = 0
		self.entry_min_extreme_threshold_sigma = 0
		self.result_entry_actual_entry_sigma = 0


