import numpy as np
import scipy as scipy
import datetime
import time
import copy

from util.memoize import memoize, MemoizeMutable

from data.redis import manage_redis
from src import math_tools

def get_corrected_data(stock_1_data, stock_2_data):
	stock_1_trimmed, stock_2_trimmed = trim_data(stock_1_data, stock_2_data)

	simple_1_close = np.empty(len(stock_1_trimmed))
	simple_2_close = np.empty(len(stock_2_trimmed))

	for k, v in enumerate(stock_1_trimmed):
		simple_1_close[k] = stock_1_trimmed[k]['AdjClose']
		simple_2_close[k] = stock_2_trimmed[k]['AdjClose']

	simple_1_returns = math_tools.get_ln_returns(simple_1_close)
	simple_2_returns = math_tools.get_ln_returns(simple_2_close)

	return simple_1_close, simple_2_close, stock_1_trimmed, stock_2_trimmed
	# return simple_1_returns, simple_2_returns


def trim_data(stock_1_data, stock_2_data):

	stock_1_start = datetime.datetime.strptime(stock_1_data[0]['Date'], '%Y-%m-%d').date()
	stock_2_start = datetime.datetime.strptime(stock_2_data[0]['Date'], '%Y-%m-%d').date()

	if stock_1_start < stock_2_start:
		# print "1 is earlier"
		for x in xrange(0, len(stock_1_data)):
			# if datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date() >= \
			#    datetime.datetime.strptime(stock_1_data[0]['Date'], '%Y-%m-%d').date():
			if stock_1_data[x]['Date'] == stock_2_data[0]['Date']:
				trim_at = x
				break
		stock_1_data = stock_1_data[trim_at:]
	
	elif stock_2_start < stock_1_start:
		# print "2 is earlier"
		for x in xrange(0, len(stock_2_data)):
			# if datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date() >= \
			#    datetime.datetime.strptime(stock_1_data[0]['Date'], '%Y-%m-%d').date():
			if stock_2_data[x]['Date'] == stock_1_data[0]['Date']:
				trim_at = x
				break
		stock_2_data = stock_2_data[trim_at:]

	if len(stock_1_data) != len(stock_2_data):
		stock_1_data, stock_2_data = propagate_on_fly(stock_1_data, stock_2_data)

	if len(stock_1_data) != len(stock_2_data) or \
		stock_2_data[len(stock_2_data)-1]['Date'] != stock_1_data[len(stock_1_data)-1]['Date'] or \
		stock_2_data[0]['Date'] != stock_1_data[0]['Date']:
		# print "\n**************"
		# print stock_1_data[0]['Symbol'], stock_2_data[0]['Symbol']
		# print len(stock_1_data), len(stock_2_data)
		# print stock_1_data[0]['Date'], stock_2_data[0]['Date']
		# print stock_1_data[len(stock_1_data)-1]['Date'], stock_2_data[len(stock_2_data)-1]['Date']
		# for x in xrange(len(stock_1_data)):
		# 		if stock_1_data[x]['Date'] != stock_2_data[x]['Date']:
		# 			print stock_1_data[x]['Date'], stock_2_data[x]['Date']
		# 			print "**************"
		# 			break
		e = stock_1_data[0]['Symbol'] + ' and ' + stock_2_data[0]['Symbol']
		raise Exception(e + ' did not trim properly and cannot be processed')

	# TODO: this section should be cythonized
	for x in xrange(len(stock_1_data)):
		if stock_1_data[x]['Date'] != stock_2_data[x]['Date']:
			e = stock_1_data[0]['Symbol'] + ' and ' + stock_2_data[0]['Symbol']
			print "\n\n"
			raise Exception(e + ' did not trim properly and cannot be processed')

	# if stock_1_end > stock_2_end:
	# 	print "1 ends later"

	# elif stock_2_end > stock_1_end:
	# 	print "2 ends later"

	return stock_1_data, stock_2_data

def propagate_on_fly(stock_1_data, stock_2_data):

	x_max = max(len(stock_1_data), len(stock_2_data))
	
	for x in xrange(0, x_max):
		
		# if x > 3397 and x < 3407:
		# 	print x, stock_1_data[x]['Date'], stock_2_data[x]['Date'], stock_1_data[x]['Close'], stock_2_data[x]['Close']
		if stock_1_data[x]['Date'] != stock_2_data[x]['Date']:
			if datetime.datetime.strptime(stock_1_data[x]['Date'], '%Y-%m-%d').date() > \
				datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date():
				temp = copy.deepcopy(stock_1_data[x-1])
				temp['Date'] = copy.deepcopy(stock_2_data[x]['Date'])
				stock_1_data.insert(x, temp)

			elif datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date() > \
				datetime.datetime.strptime(stock_1_data[x]['Date'], '%Y-%m-%d').date():
				temp = copy.deepcopy(stock_2_data[x-1])
				temp['Date'] = copy.deepcopy(stock_1_data[x]['Date'])
				stock_2_data.insert(x, temp)

	# for x in xrange(3397, 3408):
	# 	print x, stock_1_data[x]['Date'], stock_2_data[x]['Date'], stock_1_data[x]['Close'], stock_2_data[x]['Close']
		# if stock_1_data[x]['Date'] != stock_2_data[x]['Date']:
		# 	# print stock_1_data[x]['Date'], stock_2_data[x]['Date']
		# 	break

	return stock_1_data, stock_2_data


def get_paired_stock_list(stocks, fixed_stock=None):

	len_stocks = len(stocks)
	paired_list = []
	if fixed_stock is not None:
		for x in xrange(0, len_stocks):
			item = {'stock_1' : fixed_stock, 'stock_2' : stocks[x]}
			paired_list.append(item)
		print "Paired list length: ", len(paired_list)
		return paired_list

	for x in xrange(0, len_stocks):
		stock_1 = stocks[x]

		for y in xrange(x+1, len_stocks):
			stock_2 = stocks[y]

			item = {'stock_1' : stock_1, 'stock_2' : stock_2}
			paired_list.append(item)

	print "Paired list length: ", len(paired_list)
	return paired_list


def run_cointegrations():

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

	stock_list = ['XOM', 'CVX']

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
		# stock_1_data = get_data(stock=item['stock_1'])
		# stock_2_data = get_data(stock=item['stock_2'])
		stock_1_data = manage_redis.parse_fast_data(item['stock_1'])
		stock_2_data = manage_redis.parse_fast_data(item['stock_2'])
		stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)


		end_data = 10969
		end_data = len(stock_1_close)
		window_size = 400
		for x in xrange(0, end_data - window_size):
			end_index = x + window_size
				
			stock_1_window = stock_1_close[x : end_index+1]
			stock_2_window = stock_2_close[x : end_index+1]

			slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stock_1_window, stock_2_window)

			if r_value >= 0.90 and x >= 6000:
				result = show_residuals(stock_1_close, stock_2_close, x, end_index, stock_1_trimmed, stock_2_trimmed)
				print "\nIndex of price corr: ", x, end_index
				
				if result is not None:
					print "\nFound a good trade: "
					print "Start: ", stock_1_trimmed[x]['Date']
					print "End: ", stock_2_trimmed[end_index]['Date']
					out_file.write(result)
					out_file.close
					return

	out_file.close()
	print "\nFinished: ", len_pairs
	print "\nbad trim: ", bad_trim
	print "no data: ", no_data

	
	return



def show_residuals(stock_1_close, stock_2_close, start_index, end_index, stock_1_trimmed, stock_2_trimmed):
			
	stock_1_window = stock_1_close[start_index : end_index]
	stock_2_window = stock_2_close[start_index : end_index]
	
	len_stock_1_window = len(stock_1_window)

	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stock_1_window, stock_2_window)

	### Remember for the regressions: if we have this: first = [0,1,2,3,4] and second = [0,2,4,6,8] and we do:
	# linregress(first, second), we get an equation of second = slope*first + intercept
	
	residuals = []
	total_residuals = []
	smoothed_total_residuals = []
	current_residual = 0
	total_residual = 0
	spread_list = []
	output_string = ''

	price_ratio_1_2 = stock_1_close[start_index] / stock_2_close[start_index]
	hedge_ratio = slope

	print "Start: ", stock_1_close[start_index], stock_2_close[start_index], price_ratio_1_2, hedge_ratio
	print "\n\n"
	#raise Exception
	previous_position_value = 0
	for x in xrange(start_index, end_index):
		
		# placeholder in output string
		stock_2_expected = 0

		current_position_value = (hedge_ratio * stock_1_close[x]) - stock_2_close[x]
		previous_position_value = (hedge_ratio * stock_1_close[x-1]) - stock_2_close[x-1]
		current_residual = current_position_value - previous_position_value
		residuals.append(current_residual)
		total_residual += current_residual
		total_residuals.append(total_residual)


		if len(total_residuals) > 7:
			z = x - start_index
			smoothed_total_residual = np.mean(total_residuals[z-6:z+1])
			smoothed_total_residuals.append(smoothed_total_residual)
		else:
			smoothed_total_residual = 0
			smoothed_total_residuals.append(smoothed_total_residual)


		date = stock_1_trimmed[x]['Date']
		print x, date, slope, intercept, stock_1_close[x], stock_2_close[x], current_residual, total_residual

		output_string += ','.join([
								   # str(spread),
								   str(stock_1_close[x]), 
								   str(stock_2_close[x]),
								   str(stock_2_expected),
								   str(current_residual),
								   str(total_residual),
								   # str(smoothed_total_residual),
								   ]) + '\n'


	rslope, rintercept, rr_value, rp_value, rstd_err = scipy.stats.linregress(range(len_stock_1_window), total_residuals)
	
	print '\nRes slope, int, p, r: %.4f, %.4f, %.4f, %.4f' % (rslope, rintercept, rp_value, rr_value)
	print "r, slope, intercept: ", r_value, slope, intercept

	# if abs(rr_value) <= 0.3 and abs(rslope) < 0.001:
	if abs(rr_value) <= 0.003 and abs(rp_value) >= 0.95:

		import statsmodels.tsa.stattools as stats
		# see here: http://statsmodels.sourceforge.net/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller
		result = stats.adfuller(residuals)
		print result, '\n\n\n'
		raise Exception

		# we need to add to the residual list here to continue the simulation
		###################
		print "\n"
		for y in xrange(0, 500):

			i = y + end_index

			stock_2_expected = 0

			current_position_value = (hedge_ratio * stock_1_close[i]) - stock_2_close[i]
			previous_position_value = (hedge_ratio * stock_1_close[i-1]) - stock_2_close[i-1]
			current_residual = current_position_value - previous_position_value
			residuals.append(current_residual)
			total_residual += current_residual
			total_residuals.append(total_residual)


			z = len_stock_1_window + y
			smoothed_total_residual = np.mean(total_residuals[z-6:z+1])
			smoothed_total_residuals.append(smoothed_total_residual)


			date = stock_1_trimmed[i]['Date']
			print i, date, slope, intercept, stock_1_close[i], stock_2_close[i], current_residual, total_residual

			output_string += ','.join([
									   # str(spread),
									   str(stock_1_close[i]), 
									   str(stock_2_close[i]),
									   str(stock_2_expected),
									   str(current_residual),
									   str(total_residual),
									   # str(smoothed_total_residual)
									   ]) + '\n'




		print "\nMean / Sigma of total resid: ", np.mean(total_residuals), np.std(total_residuals)
		####################
		return output_string
	else:
		return None





















# def show_residuals(stock_1_close, stock_2_close, start_index, end_index):
			
# 	stock_1_window = stock_1_close[start_index : end_index]
# 	stock_2_window = stock_2_close[start_index : end_index]
	
# 	len_stock_1_window = len(stock_1_window)

# 	# print "Len: ", len_stock_1_window, (end_index - start_index), start_index, end_index
# 	# print stock_1_window
# 	# print stock_2_window

# 	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stock_1_window, stock_2_window)

# 	### Remember for the regressions: if we have this: first = [0,1,2,3,4] and second = [0,2,4,6,8] and we do:
# 	# linregress(first, second), we get an equation of second = slope*first + intercept
	
# 	residuals = []
# 	current_residual = 0
# 	total_residual = 0
# 	spread_list = []
# 	output_string = ''

# 	for x in xrange(start_index, end_index):
		
# 		stock_2_expected = (slope * stock_1_close[x]) + intercept
# 		current_residual = stock_2_expected - stock_2_close[x]
# 		total_residual += current_residual
# 		residuals.append(current_residual)

# 		print x, slope, intercept, stock_1_close[x], stock_2_close[x], stock_2_expected, current_residual, total_residual

# 		output_string += ','.join([
# 								   # str(spread),
# 								   str(stock_1_close[x]), 
# 								   str(stock_2_close[x]),
# 								   str(stock_2_expected),
# 								   str(current_residual),
# 								   str(total_residual),
# 								   ]) + '\n'


# 	rslope, rintercept, rr_value, rp_value, rstd_err = scipy.stats.linregress(range(len_stock_1_window), residuals)
	
# 	print '\nRes slope, int, p, r: %.4f, %.4f, %.4f, %.4f' % (rslope, rintercept, rp_value, rr_value)
# 	print "r, slope, intercept: ", r_value, slope, intercept

# 	# if abs(rr_value) <= 0.3 and abs(rslope) < 0.001:
# 	if abs(rr_value) <= 0.003 and abs(rp_value) >= 0.95:

# 		# we need to add to the residual list here to continue the simulation
# 		###################
		
# 		print "\n"
# 		for y in xrange(0, 50):

# 			i = y + end_index

# 			stock_2_expected = (slope * stock_1_close[i]) + intercept
# 			current_residual = stock_2_expected - stock_2_close[i]
# 			total_residual += current_residual
# 			residuals.append(current_residual)

# 			print i, slope, intercept, stock_1_close[i], stock_2_close[i], stock_2_expected, current_residual, total_residual

# 			output_string += ','.join([
# 									# str(spread),
# 									str(stock_1_close[i]), 
# 									str(stock_2_close[i]),
# 									str(stock_2_expected),
# 									str(current_residual),
# 									str(total_residual),
# 									]) + '\n'




# 		####################
# 		return output_string
# 	else:
# 		return None

	# print "\nCorr Coeff: ", np.corrcoef(stock_1_window, stock_2_window)[0][1]

	# print "\n\nResidual Slope: ", rslope
	# print "Residual Intercept: ", rintercept
	# print "Residual R Value: ", rr_value
	# print "Residual P Value: ", rp_value
	# print "Residual Std Err: ", rstd_err

	# print "\n\nSlope: ", slope
	# print "Intercept: ", intercept
	# print "R Value: ", r_value
	# print "P Value: ", p_value
	# print "Std Err: ", std_err
