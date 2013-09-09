import numpy as np
import scipy as scipy
import datetime
import time

from util.memoize import memoize, MemoizeMutable

from data.redis import manage_redis
# from src import math_tools
from src.cointegrations_data import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly


def run_cointegrations():
	"""
	Takes in data from stocks, and performs a basic ADF test on the data.
	"""

	sectors = ('basic_materials', 'conglomerates', 'consumer_goods', 'financial', 'healthcare', 'industrial_services', \
			   'services', 'technology', 'utilities')

	# location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/list_sp_500.csv'
	# location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/sectors/technology.csv'
	# location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/300B_1M.csv'
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/300B_1M_and_etfs_etns.csv'

	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	for k, item in enumerate(stock_list):
		new_val = item.split('\r')[0]
		stock_list[k] = new_val
	in_file.close()

	stock_list = ['GS', 'JPM']

	paired_list = get_paired_stock_list(stock_list, fixed_stock=None)

	good_corr_data = []
	corr_list = []
	beta_list = []
	no_data = 0
	bad_trim = 0
	len_pairs = len(paired_list)

	interest_items = []

	out_file = open('/home/wilmott/Desktop/fourseasons/fourseasons/cointegration_results.csv', 'w')

	time_start = time.time()

	for k, item in enumerate(paired_list):
		stock_1_data = manage_redis.parse_fast_data(item['stock_1'])
		stock_2_data = manage_redis.parse_fast_data(item['stock_2'])
		stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)

		end_data = len(stock_1_close)
		window_size = 400
		for x in xrange(0, end_data - window_size):
		# for x in xrange(0, 12000):
			end_index = x + window_size
				
			stock_1_window = stock_1_close[x : end_index+1]
			stock_2_window = stock_2_close[x : end_index+1]

			slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stock_1_window, stock_2_window)
			lin_reg_results = (slope, intercept, r_value, p_value, std_err)

			# Here, we will only consider stocks where we can first see that a linreg of their prices can be done with
			# a high r-value. If we can't even find a good correlation between the stocks' prices, we skip. This is like
			# a sanity check - we should not be trying to trade 'MSFT' and 'XOM' as a pair trade!!!
			if r_value >= 0.90:
				result = create_synthetic_investment(stock_1_close, stock_2_close, x, end_index, stock_1_trimmed, \
													 stock_2_trimmed, lin_reg_results)
				# print "Window of high correlation used for prev. ADF Test: ", x, end_index
				if result is not None:
					print "\nFound a good window: "
					print "Start: ", stock_1_trimmed[x]['Date']
					print "End: ", stock_2_trimmed[end_index]['Date']
					out_file.write(result)
					out_file.close
					return
					pass
			else:
				# print "PRICES NOT CORRELATED", x
				pass

	out_file.close()
	print "\nFinished: ", len_pairs
	print "\nbad trim: ", bad_trim
	print "no data: ", no_data

	
	return



def create_synthetic_investment(stock_1_close, stock_2_close, start_index, end_index, stock_1_trimmed, \
							    stock_2_trimmed, lin_reg_results):
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
	previous_position_value = 0
	for x in xrange(start_index, end_index):

		# adding intercept simply corrects for an offset that would otherwise exist when the data is output
		synthetic_price = (hedge_ratio * stock_1_close[x]) - stock_2_close[x] + intercept
		synthetic_prices.append(synthetic_price)

		synthetic_price_pc = synthetic_price / ((hedge_ratio * stock_1_close[start_index]) + stock_2_close[start_index] + intercept)
		synthetic_prices_pc.append(synthetic_price_pc)

		date = stock_1_trimmed[x]['Date']
		### print x, date, slope, intercept, stock_1_close[x], stock_2_close[x], current_residual, total_residual

		output_string += ','.join([
								   # str(spread),
								   str(stock_1_close[x]), 
								   str(stock_2_close[x]),
								   str(0),
								   str(synthetic_price_pc),
								   str(0),
								   # str(smoothed_total_residual)
								   ]) + '\n'
		## print x, date, round(total_residual, 4), round(total_residual_pc, 6), '%'


	rslope, rintercept, rr_value, rp_value, rstd_err = scipy.stats.linregress(range(len_stock_1_window), \
																					synthetic_prices_pc)
	
	# print '\nTotal residuals slope, int, p, r: %.4f, %.4f, %.4f, %.4f' % (rslope, rintercept, rp_value, rr_value)
	# print 'We should expect slope near zero, low r value wrt time and low p value to indicate it is not random chance'
	# print '\nStock prices r, slope, intercept: %.4f, %.4f, %.4f' % (r_value, slope, intercept)
	
	# We first run a sanity check before even attempting a DF test - here, we verify that the r value of the synthetic
	# investment is somewhat low - it should be, indicating that it is not particularly correlated to the time. 
	if abs(rr_value) <= 0.5:
		# print 'Synth inv: %i %.4f %.4f %.4f %.4f %.4f' % (start_index, rslope, rintercept, rr_value, rp_value, rstd_err)
		result = do_df_test(synthetic_prices_pc)
	else:
		# print 'SKIPPED: ', start_index
		return None

	if result and result[0] <= result[4]['5%'] and result[1] < 0.1:
		print start_index, stock_1_trimmed[start_index]['Date'], result[0], result[4]['5%'], result[1], \
			  round(slope, 4), round(intercept, 4)
		
		mean_price = round(np.mean(synthetic_prices_pc), 5)
		sigma_price = round(np.std(synthetic_prices_pc), 5)
		if sigma_price < 0.05:
			return None

		print "Synth investment stats: ", mean_price, sigma_price, '\n'
		return output_string
	elif result:
		# print "FAILED DF TEST: ", start_index, result[0], result[4]['5%'], result[1], round(slope, 4), round(intercept, 4)
		pass
	
	return None



def do_df_test(prices):

	import statsmodels.tsa.stattools as stats
	# see here: http://statsmodels.sourceforge.net/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller
	# result is in the format: test statistic (lambda), p_value, used lag, # of observations, critical values
	result = stats.adfuller(prices)
	return result




### I had an idea to use smoothing on the residuals but I don't think this makes sense...
# if len(total_residuals) > 7:
# 	z = x - start_index
# 	smoothed_total_residual = np.mean(total_residuals[z-6:z+1])
# 	smoothed_total_residuals.append(smoothed_total_residual)
# else:
# 	smoothed_total_residual = 0
# 	smoothed_total_residuals.append(smoothed_total_residual)
