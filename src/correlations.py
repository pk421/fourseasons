from src.data_retriever import read_redis
import numpy as np
import toolsx as tools
import datetime
import copy

from util.profile import profile
from util.memoize import memoize, MemoizeMutable

from data.redis import manage_redis


#@MemoizeMutable
def get_data(stock=None):

	if stock:
		return read_redis(stock=stock, db_number=0, dict_size=10)[0]

def get_corr_coeff(stock_1_data, stock_2_data):

	stock_1_trimmed, stock_2_trimmed = trim_data(stock_1_data, stock_2_data)

	simple_1_close = np.empty(len(stock_1_trimmed))
	simple_2_close = np.empty(len(stock_2_trimmed))

	for k, v in enumerate(stock_1_trimmed):
		simple_1_close[k] = stock_1_trimmed[k]['AdjClose']
		simple_2_close[k] = stock_2_trimmed[k]['AdjClose']

	corr_coeff = np.corrcoef(simple_1_close, simple_2_close)[0][1]
	cov_coeff = np.cov(simple_1_close, simple_2_close)
	variance_1 = cov_coeff[0][0]
	variance_2 = cov_coeff[1][1]
	covariance = cov_coeff[0][1]
	beta = covariance / variance_1
	# np.cov() will return a covariance matrix. upper left is the variance of the first arg, upper right is covariance
	# of the two, lower left is covariance of the two (same as upper right), lower right is the variance of 2nd arg
	# benchmark (SPY) comes in as stock 1 here.
	
	# print np.corrcoef(simple_1_close, simple_2_close), cov_coeff
	return corr_coeff, beta

#@MemoizeMutable
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
			item = {'stock_1' : fixed_stock, 'stock_2' : stocks[x], 'corr_coeff' : 0.0, 'beta' : 0.0}
			paired_list.append(item)
		print "Paired list length: ", len(paired_list)
		return paired_list

	for x in xrange(0, len_stocks):
		stock_1 = stocks[x]

		for y in xrange(x+1, len_stocks):
			stock_2 = stocks[y]

			item = {'stock_1' : stock_1, 'stock_2' : stock_2, 'corr_coeff' : 0.0, 'beta' : 0.0}
			paired_list.append(item)

	print "Paired list length: ", len(paired_list)
	return paired_list


#@profile
def run_correlations():
	# test_redis()
	# fill_redis()
	# return

	sectors = ('basic_materials', 'conglomerates', 'consumer_goods', 'financial', 'healthcare', 'industrial_services', \
			   'services', 'technology', 'utilities')

	#location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/list_sp_500.csv'
	#location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/300B_1M.csv'
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/sectors/technology.csv'
	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	for k, item in enumerate(stock_list):
		new_val = item.split('\r')[0]
		stock_list[k] = new_val
	in_file.close()

	# stock_list = ['HPQ', 'GRMN']

	paired_list = get_paired_stock_list(stock_list, fixed_stock='SPY')

	good_corr_data = []
	corr_list = []
	beta_list = []
	no_data = 0
	bad_trim = 0
	len_pairs = len(paired_list)

	out_file = open('/home/wilmott/Desktop/fourseasons/fourseasons/correlation_results.csv', 'w')

	for k, item in enumerate(paired_list):
		# stock_1_data = get_data(stock=item['stock_1'])
		# stock_2_data = get_data(stock=item['stock_2'])
		stock_1_data = manage_redis.parse_data(item['stock_1'])
		stock_2_data = manage_redis.parse_data(item['stock_2'])

		if stock_1_data is None or len(stock_1_data) == 0:
			item['corr_coeff'] = -8888
			print "\n", k, len_pairs, item['corr_coeff'], item['beta'], item['stock_1'], "ERROR: No Data"
			no_data += 1
			continue
		if stock_2_data is None or len(stock_2_data) == 0:
			item['corr_coeff'] = -8888
			print "\n", k, len_pairs, item['corr_coeff'], item['beta'], item['stock_2'], "ERROR: No Data"
			no_data += 1
			continue

		try:
			item ['corr_coeff'], item['beta'] = get_corr_coeff(stock_1_data, stock_2_data)
			corr_list.append(item['corr_coeff'])
			beta_list.append(item['beta'])
		except:
			# this is intended to indicate an error condition, usually the -9999 will indicate that the stocks
			# could not be trimmed properly together.
			item['corr_coeff'] = -9999
			bad_trim += 1
			print k, len_pairs, item['corr_coeff'], item['beta'], item['stock_1'], item['stock_2']
			continue

		out_string = item['stock_1'] +','+ item['stock_2'] +','+ str(item['corr_coeff']) + str(item['beta']) + '\n'
		out_file.write(out_string)
		good_corr_data.append(item)
		print k, len_pairs, item['corr_coeff'], item['beta'], item['stock_1'], item['stock_2']	


	out_file.close()
	print "\nFinished: ", len_pairs
	print "Average Beta: ", np.mean(beta_list)
	print "Sigma Beta: ", np.std(beta_list)
	print "\nbad trim: ", bad_trim
	print "good corr data: ", len(good_corr_data)
	print "no data: ", no_data
	
	return

def fill_redis():
	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/300B_1M.csv'
	in_file = open(location, 'r')
	stock_list = in_file.read().split('\r')
	in_file.close()

	import redis
	redis_writer = redis.StrictRedis(host='localhost', port=6379, db=3)

	for x in xrange(0, len(stock_list)):
		stock_list[x] = stock_list[x].strip('\n')
	# print stock_list

	for stock in stock_list:
		raw_read = read_redis(stock=stock, db_number=0, dict_size=10)[0]
		st_version = str(raw_read)
		# print st_version
		input_key = 'historical:fast:' + stock
		redis_writer.set(input_key, st_version)
		print stock, " Loaded"


def test_redis():

	location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/300B_1M.csv'
	in_file = open(location, 'r')
	stock_list = in_file.read().split('\n')
	in_file.close()

	# for k, item in enumerate(stock_list[0:50]):
	# 	res = get_data(stock=item.split()[0])
	# 	print k, item

	import redis
	redis_db = redis.StrictRedis(host='localhost', port=6379, db=15)


	#To simulate AA do this 12908 times
	# 2013-04-12,8.3,8.32,8.17,8.22,21450400.0,8.3,8.32,8.17,8.22
	
	r2 = redis.StrictRedis(host='localhost', port=6379, db=0)
	AAResults = read_redis(stock='AA', db_number=0, dict_size=10)[0]
	AAstr = str(AAResults)

	s58 = "2013-04-12,8.3,8.32,8.17,8.22,21450400.0,8.3,8.32,8.17,8.22"
	s14 = "2013-04-12,8.3"
	s1 = "1"
	x_max = 10
	redis_db.set('testKey', AAstr)

	import time
	start = time.time()
	for x in xrange(0, x_max):
		res = redis_db.get('testKey')
		#print x, res
		output_data = manage_redis.parse_data(res)
	
	print "type: ", type(output_data)

	redis_db.flushdb()
	print "Finished redis test"
	end = time.time()
	print "time in gets: ", end-start

	return

# def parse_data(stock_data):

# 	output_data = []

#  	all_days = stock_data.split('}, {')
#  	print len(all_days)
#  	for day in all_days:
#  		day = day.strip('{}[]')

#  		keys = []
#  		values = []
#  		for e in day.split(', '):
#  			# print day
#  			# print "E is: ", e.split(': ')
#  			k = e.split(': ')
# 			keys.append(k[0].strip('\''))
# 			# print "K0: ", k[0]
# 			try:
# 				values.append(float(k[1]))
# 			except:
# 				values.append(k[1].strip('\''))

# 		todays_dict = dict(zip(keys, values))
# 		output_data.append(todays_dict)
# 		# print "Today's Dict: ", todays_dict

# 	return output_data
