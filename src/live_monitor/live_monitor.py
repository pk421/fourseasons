import datetime
import time
import requests

import redis

from include import kronos

from src.data_retriever import load_redis
from src.data_retriever import multithread_yahoo_download
from data.redis import manage_redis

import src.toolsx as tools

import numpy as np

from src.cointegrations_data import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly, \
									get_bunches_of_pairs



def reset_symbol_list_key():
	in_file_name = 'etfs_etns'
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + in_file_name + '.csv'

	stock_list = []
	in_file = open(location, 'r')
	input_string = in_file.read().split('\n')
	for k, item in enumerate(input_string):
		new_val = item.split('\r')[0]
		if new_val == '':
			continue
		stock_list.append(new_val)
	in_file.close()


	input_string = ','.join(sorted(stock_list))
	redis_writer = redis.StrictRedis(host='localhost', port=6379, db=14)
	redis_writer.set('symbol_list', input_string)

	### redis_writer.rpush('symbols_with_updated_data', stock_list)
	redis_writer.delete('symbols_with_updated_data')

	return

def download_historical_data():

	redis_reader = redis.StrictRedis(host='localhost', port=6379, db=14)
	master_keys = redis_reader.get('symbol_list').split(',')

	already_updated = redis_reader.smembers('symbols_with_updated_data')

	symbols_to_get = []

	for symbol in master_keys:
		if symbol in already_updated:
			continue
		symbols_to_get.append(symbol)

	print "start downloading"
	multithread_yahoo_download(thread_count=20, update_check=False, \
                               new_only=False, store_location = 'data/live_system/', use_list=master_keys)
	print "finished downloading"

	load_redis(stock_list='tda_free_etfs.csv', db_number=14, file_location='data/live_system/', dict_size=3, use_list=master_keys)
	print "finished loading redis"

	return

def do_web_query(symbol_string, retry=5):
	count = 0
	while count < retry:
		try:
			# See here for keys: http://www.gummy-stuff.org/Yahoo-data.htm
			query_string = 'http://finance.yahoo.com/d/quotes.csv?s=' + symbol_string + '&f=sl1d1t1k1a2'
			data = requests.get(query_string, timeout=10).text
			return True, data
		except:
			count += 1
	return False, ''



def search_for_trades(in_trade=[]):

	"""Algo: Try to do a batch request to get the latest quotes for all symbols from yahoo. Do a GET from redis for each
	stock, append the latest price to the end of the list, then run thru MACD / RSI. See if stock meets criteria,
	if it does, append it to a list of interesting stocks and send email.

	"""
	redis_reader = redis.StrictRedis(host='localhost', port=6379, db=14)
#	symbols = redis_reader.get('symbol_list').split(',')
	symbols = redis_reader.get('symbol_list').split(',')

	query_strings = []
	start_pos = 0
	end_pos = 200
	len_symbols = len(symbols)
	while True:
		if end_pos >= len_symbols:
			query_strings.append('+'.join(symbols[start_pos:]))
			break
		else:
			query_strings.append('+'.join(symbols[start_pos:end_pos]))
			start_pos += 200
			end_pos += 200

#	print len(query_strings)
#	print type(query_strings[0])
#	print len(query_strings[7])
#	print query_strings[7]
#
#	raise Exception




	### symbol_string = '+'.join(symbols)

#	try:
#		# query_string = 'http://finance.yahoo.com/d/quotes.csv?s=' + symbol_string + '&f=sl1t1d1k1'
#		query_string = 'http://finance.yahoo.com/d/quotes.csv?s=' + symbol_string + '&f=sl1d1t1k1a2'
#		data = requests.get(query_string, timeout=20).text
#	except:
#		print "Web data query failed."
#		try:
#			subject_to_use = 'TS Results: Failure ' +  str(datetime.datetime.now().time())
#			send_email(subject=subject_to_use, body='Web Data Query Failed')
#		except:
#			pass
#		return None


	d = []
	for x in xrange(0, len(query_strings)):
		print x
		result, data = do_web_query(query_strings[x])
		print "RESULT:", result
		print "DATA:", data

		d.append(data)
		print len(d), result, len(query_strings)

		if result != True:
			print "Web data query failed."
			try:
				subject_to_use = 'TS Results: Failure ' +  str(datetime.datetime.now().time())
				send_email(subject=subject_to_use, body='Web Data Query Failed')
			except:
				return None

	data = ''.join(d)
	print data

	current_price_dict = {}
	current_date_dict = {}
	current_trade_time_dict = {}
	rt_time_dict = {}
	average_volume_dict = {}

	###
	print data
	prices = data.split('\r\n')

	for p in prices:
		items = p.split(',')
		if len(items) > 3:

			try:
				if int(str(items[5]).strip('\"')) < 100000:
					# this filter cuts out items that have a small Volume
					continue
			except:
				print "Could not filter by using items[5] for volume: "
				print items
				continue

			s = str(items[0]).strip('\"')
			current_price_dict[s] = float(items[1])
			current_date_dict[s] = str(items[2]).strip('\"')
			current_trade_time_dict[s] = str(items[3]).strip('\"')
			rt_time_dict[s] = str(items[4]).strip('\"')
			average_volume_dict[s] = str(items[5]).strip('\"')

	output_data = []

	### print current_price_dict, current_date_dict

	for symbol in current_price_dict:
		# Here we must use the current_price_dict and NOT "symbols" because current_price_dict reflects the actual data
		# That was obtained and is available, rather than simply the stored symbol that we attempted to retrieve
		latest_price = current_price_dict[symbol]
		latest_date = current_date_dict[symbol]
		latest_time = current_trade_time_dict[symbol]
		latest_rt = rt_time_dict[symbol]
		volume = average_volume_dict[symbol]
		ret_code, result = get_parameters(symbol, latest_price, latest_date, latest_time, latest_rt, volume, in_trade)

		if ret_code or (symbol in in_trade):
			output_data.append(result)

	
	###FIXME: This sorted_output line fails sometimes...why?
	# use this item to sort on the RSI b/c of the 50-
	#sorted_output = sorted(output_data, key=lambda item: abs(50-item[6]), reverse=True)
	sorted_output = sorted(output_data, key=lambda item: abs(float(item[5])), reverse=True)

	at_top = []
	for item in sorted_output:
		if item[0] in in_trade:
			at_top.append(item)
	at_top.extend(sorted_output)
	sorted_output = at_top

	# (symbol, latest_date, latest_time, volume, latest_price, sigma_over_p_0, rsi_0, sma_0, sigma_0, stop_loss_offset, latest_rt)
	body = '\t'.join(['Symbol', 'Trade Date', 'Trade Time', 'Volume', 'Price', 'Sig/P','RSI', 'SMA', 'Sigma', 'SL Offset', 'Trade RT']) + '\n\n'
	for item in sorted_output:
		new_item = [str(a).rjust(11, ' ') for a in item]
		data1 = ''.join(new_item[:5])
		# cut out leading spaces before the symbol line
		data1 = data1[7:]
		# pad in leading spaces in the second chunk so everything lines up indented
		data2 = '    ' + ''.join(new_item[5:])

		body += data1 + '\n' + data2 + '\n\n'


	print body
	
	subject_to_use = 'TS Results: ' + str(len(sorted_output)) + '   ' + str(datetime.datetime.now().time())

	try:
		send_email(subject=subject_to_use, body=body)
	except:
		pass

	return


def get_parameters(symbol, latest_price, latest_date, latest_time, latest_rt, volume, in_trade=[]):
	
	stock_1_data = manage_redis.parse_fast_data('TLT', db_to_use=14)
	stock_2_data = manage_redis.parse_fast_data(symbol, db_to_use=14)
	try:
		# print "Getting data for: ", 'TLT', symbol
		stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
	except:
		print 'errors found in historical data ' + symbol
		return False, 'errors found in historical data ' + symbol

	current_date_normalized = datetime.datetime.strptime(latest_date, '%m/%d/%Y')
	historical_date_normalized = datetime.datetime.strptime(stock_2_trimmed[-1]['Date'], '%Y-%m-%d')

	### print symbol, current_date_normalized, historical_date_normalized

#	if not current_date_normalized > historical_date_normalized:
#		print 'current_date is not greater than historical_date for ' + symbol
#		return False, 'current_date is not greater than historical_date for ' + symbol
#		pass

	if len(stock_2_close) < 205:
		return False, "not enough historical data for " + symbol

	### merged_prices = stock_2_close
	merged_prices = np.append(stock_2_close, latest_price)

	rsi = tools.rsi(merged_prices, 4)
	rsi_0 = round(rsi[-1], 4)
	rsi_1 = round(rsi[-2], 4)
	sma_0 = round(tools.simple_moving_average(merged_prices, 200)[-1], 4)
	sigma_0 = round(np.std(merged_prices[-101:]), 4)
	sigma_over_p_0 = round(sigma_0 / float(latest_price), 4)

	###
	# print "Symbol: ", symbol, merged_prices[-5:], rsi_0, rsi_1, sma_0, sigma_0, sigma_over_p_0

	entry_bound = 25
	exit_bound = 100-entry_bound
	minimum_vol = 0.00

	if symbol in in_trade:
		stop_loss_offset = round(1.4 * sigma_0, 4)
		return True, (symbol, latest_date, latest_time, volume, latest_price, sigma_over_p_0, rsi_0, sma_0, sigma_0, stop_loss_offset, latest_rt)

	if sigma_over_p_0 >= minimum_vol:
		if latest_price > sma_0:
			if (rsi_1 > entry_bound and rsi_0 < entry_bound):
				stop_loss_offset = round(1.4 * sigma_0, 4)
				return True, (symbol, latest_date, latest_time, volume, latest_price, sigma_over_p_0, rsi_0, sma_0, sigma_0, stop_loss_offset, latest_rt)

		else:
			if (rsi_1 < exit_bound and rsi_0 > exit_bound):
				stop_loss_offset = round(1.4 * sigma_0, 4)
				return True, (symbol, latest_date, latest_time, volume, latest_price, sigma_over_p_0, rsi_0, sma_0, sigma_0, stop_loss_offset, latest_rt)



	return False, None

 
def send_email(subject='Trade Scan Results', body='Test Body'):

	import smtplib

	from_address = 'fourseasonsnoreply@dontreply.com'
	to_address = 'mcpilat@gmail.com'

	msg = "\r\n".join([
		"From: fourseasonsnoreply@dontreply.com",
		"To: mcpilat@gmail.com",
		"Subject: " + subject,
		body
		])

	username = 'fourseasonsnoreply'
	password = '4seasons'

	server = smtplib.SMTP('smtp.gmail.com:587')
	server.starttls()
	server.login(username,password)
	server.sendmail(from_address, to_address, msg)
	server.quit()

	print "Finished sending"

def run_live_monitor():

	reset_symbol_list_key()
	
	### load symbol list from redis, store as a temporary csv, then use it for the download / uupdate functions

	# scheduler = kronos.Scheduler()
	
	# scheduler.add_interval_task(search_for_trades,
	# 							'search_for_trades',
	# 							0,
	# 							180,
	# 							kronos.method.sequential,
	# 							[],
	# 							{})

	
	# scheduler.start()
	download_historical_data()
	### search_for_trades()
	# send_email()

	while(True):
		time.sleep(0.25)
		current_time = datetime.datetime.now()
		current_hour = current_time.hour
		current_minute = current_time.minute
		current_second = current_time.second
		print current_time

		if current_hour >= 0 and current_hour <= 6:
			if current_minute % 30 == 0:
				try:
					download_historical_data()
					time.sleep(600)
				except:
					print "Exception in the download process!!"
					time.sleep(60)
					continue
		elif (current_hour == 7 and current_minute < 30):
			search_for_trades()
			time.sleep(600)
		elif (current_hour >= 9) and (current_hour <= 16) and not (current_hour == 15 and current_minute >= 45):
			if current_minute % 30 == 0:
				search_for_trades()
				time.sleep(600)
		elif (current_hour == 15 and current_minute > 45):
			search_for_trades()
			time.sleep(40)

		# elif (current_hour == 19 and current_minute > 45):
		# 	search_for_trades()
		# 	time.sleep(50)


		time.sleep(10)





	return


