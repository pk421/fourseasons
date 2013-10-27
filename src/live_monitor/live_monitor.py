import datetime
import time
import requests

import redis

from src.data_retriever import load_redis
from src.data_retriever import multithread_yahoo_download



def reset_symbol_list_key():
	in_file_name = 'tda_free_etfs'
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/sectors/' + in_file_name + '.csv'

	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	for k, item in enumerate(stock_list):
		new_val = item.split('\r')[0]
		stock_list[k] = new_val
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

	# multithread_yahoo_download(thread_count=20, update_check=False, \
 #                               new_only=False, store_location = 'data/live_system/', use_list=symbols_to_get)

	load_redis(stock_list='sectors/tda_free_etfs.csv', db_number=14, file_location='data/live_system/', dict_size=2, use_list=symbols_to_get)

	return


def search_for_trades():

	"""Algo: Try to do a batch request to get the latest quotes for all symbols from yahoo. Do a GET from redis for each
	stock, append the latest price to the end of the list, then run thru MACD / RSI. See if stock meets criteria,
	if it does, append it to a list of interesting stocks and send email.

	"""
	redis_reader = redis.StrictRedis(host='localhost', port=6379, db=14)
	symbols = redis_reader.get('symbol_list').split(',')
	symbol_string = '+'.join(symbols)

	query_string = 'http://finance.yahoo.com/d/quotes.csv?s=' + symbol_string + '&f=sl1t1k1'
	data = requests.get(query_string).text

	current_price_dict = {}

	print data
	prices = data.split('\r')
	print prices
	return
	for p in prices:
		items = p.split(',')
		current_price_dict[items[0]] = items[1]

	print current_price_dict


 
def send_email():

	import smtplib

	from_address = 'fourseasonsnoreply@dontreply.com'
	to_address = 'mcpilat@gmail.com'

	msg = "\r\n".join([
		"From: fourseasonsnoreply@dontreply.com",
		"To: mcpilat@gmail.com",
		"Subject: Test EMail",
		"Test body"
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

	#reset_symbol_list_key()
	
	### load symbol list from redis, store as a temporary csv, then use it for the download / uupdate functions
	#download_historical_data()

	search_for_trades()

	#send_email()

	return


