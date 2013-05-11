# import pyximport
# pyximport.install()

from src.data_retriever import read_redis
import numpy as np
import scipy.stats.mstats as sp
import datetime


import toolsx as tools

def run_harding_seasonality():
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/300B_1M.csv'
	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	in_file.close()

	# stock_list = ['AAPL', 'GOOG', 'GLD', 'SLV', 'NEM', 'XOM', 'CVX']
	stock = 'SPY'

	sys_returns = np.empty(len(stock_list))
	buy_hold_returns = np.empty(len(stock_list))
	failed = []

	for k, v in enumerate(stock_list):	
		v = v.split('\r')[0]	#necessary to correct some lists
		try:
			sys_returns[k], buy_hold_returns[k] = analyze_seasonality(v)
		except:
			sys_returns[k], buy_hold_returns[k] = 1, 1
			failed.append(v)
			continue
		print k, v, round(sys_returns[k], 4), round(buy_hold_returns[k], 4)

	print "\nArith Mean Annual Sys Returns: ", round(np.mean(sys_returns), 4)
	print "Arith Mean Annual B/H Returns: ", round(np.mean(buy_hold_returns), 4)
	print "\nFailures: ", failed

	return




# def analyze_seasonality(str stock):
def analyze_seasonality(stock):

	stock_data = read_redis(stock=stock, db_number=0, to_disk=False)[0]

	len_data = len(stock_data)
	simple_close_data = np.empty(len_data)
	y = np.empty(len_data)
	m = np.empty(len_data)
	d = np.empty(len_data)

	#get the date data into datetime objects in a list parallel to the simple_price_data
	for k, v in enumerate(stock_data):
		simple_close_data[k] = v['Close']
		
		date_components = v['Date'].split('-')
		y[k],m[k],d[k] = int(date_components[0]), int(date_components[1]), int(date_components[2])

	sell_trig_date, sell_window, buy_trig_date, buy_window = get_date_windows(len_data, y, m, d) 

	annual_returns = []
	trade_count = 0
	in_market = False
	entry_price = 0
	start_trade_index = 0	# For now we set it to start the trading season in June, so that the first trade is a buy
	
	for x in xrange(1, len_data):
		if m[x] == 6 and m[x-1] == 5:
			start_trade_index = x
			break


	# sma = tools.simple_moving_average(simple_close_data, 50)
	macd_line, signal_line = tools.macd(stock_data, 12, 26, 9)

	#This starts on the first day trading day in June
	for x in xrange(start_trade_index, len_data):
		date = stock_data[x]['Date']
		close = stock_data[x]['Close'] 

		#Buying Triggers
		if in_market == False:

			if buy_trig_date[x]:
				in_market = True
				entry_price = close
				# print "\nBuy: ", date, close

			# if buy_window[x] and macd_line[x] > signal_line[x] and macd_line[x-1] < signal_line[x-1]:
			# 	in_market = True
			# 	entry_price = close
			# 	print "\nBuy: ", date, close

			# elif buy_trig_date[x] and not buy_trig_date[x-1] and macd_line[x] > signal_line[x]:
			# 	in_market = True
			# 	entry_price = close
			# 	print "\nBuy: ", date, close

			# elif buy_window[x-1] and not buy_window[x]:
			# 	in_market = True
			# 	entry_price = close
			# 	print "\nBuy: ", date, close

		#Selling Triggers
		if in_market == True:

			if sell_trig_date[x]:
				in_market = False
				trade_count += 1
				gain = close / entry_price
				annual_returns.append(gain)
				# print "Sell: ", date, gain


			# if sell_window[x] and macd_line[x] < signal_line[x] and macd_line[x-1] > signal_line[x-1]:
			# 	in_market = False
			# 	trade_count += 1
			# 	gain = close / entry_price
			# 	annual_returns.append(gain)
			# 	print "Sell: ", date, gain

			# elif sell_trig_date[x] and not sell_trig_date[x-1] and macd_line[x] < signal_line[x]:
			# 	in_market = False
			# 	trade_count += 1
			# 	gain = close / entry_price
			# 	annual_returns.append(gain)
			# 	print "Sell: ", date, gain

			# elif sell_window[x-1] and not sell_window[x]:
			# 	in_market = False
			# 	trade_count += 1
			# 	gain = close / entry_price
			# 	annual_returns.append(gain)
			# 	print "Sell: ", date, gain


	final_value = np.prod(annual_returns)
	# print "\nTotal Trades: ", trade_count

	# print "\nFinal Value: ", final_value
	# print "Geometric Mean Ret: ", sp.gmean(annual_returns)

	overall_gain = (gain * entry_price) / stock_data[start_trade_index]['Close']
	# print "\nBuy and Hold Return: ", overall_gain
	# print "Buy and Hold Mean GMean Ret: ", overall_gain**(1.0/trade_count)


	return sp.gmean(annual_returns), overall_gain**(1.0/trade_count)






# cdef get_date_windows(int len_data, y, m, d):
def get_date_windows(len_data, y, m, d):

	sell_window = np.empty(len_data)
	buy_window = np.empty(len_data)
	sell_trig_date = np.empty(len_data)
	buy_trig_date = np.empty(len_data)

	sell_window[0] = False
	buy_window[0] = False
	sell_trig_date[0] = False
	buy_trig_date[0] = False

	for x in xrange(1, len_data):

		if m[x] == 5 and m[x-1] == 4:
			sell_trig_date[x] = True
		else:
			sell_trig_date[x] = False

		if m[x] == 11 and m[x-1] == 10:
			buy_trig_date[x] = True
		else:
			buy_trig_date[x] = False


		if (m[x] == 4 and d[x] >= 15) or \
			(m[x] == 5 and d[x] < 15):
			sell_window[x] = True
		else:
			sell_window[x] = False

		if (m[x] == 10 and d[x] >= 15) or \
			(m[x] == 11 and d[x] < 15):
			buy_window[x] = True
		else:
			buy_window[x] = False

		# print y[x], m[x], d[x]
		# print sell_trig_date[x], sell_window[x], buy_trig_date[x], buy_window[x]

	return sell_trig_date, sell_window, buy_trig_date, buy_window
 
	