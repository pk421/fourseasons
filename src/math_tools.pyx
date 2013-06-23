from util.memoize import memoize, MemoizeMutable

import math

@MemoizeMutable
def get_ln_returns(stock_close):

	len_data = len(stock_close)
	stock_ln_returns = []

	# print len_data, type(stock_ln_returns), stock_close
	for x in xrange(1, len_data):
		ret = math.log(float(stock_close[x]) / float(stock_close[x-1]))
		# ret = float(stock_close[x]) / float(stock_close[x-1])
		stock_ln_returns.append(ret)

	return stock_ln_returns