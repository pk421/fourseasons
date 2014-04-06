
import src.toolsx as tools
import math

class SignalsBase(object):

	def __init__(self, closes, volume, stock_2_trimmed, item):
		self.closes = closes
		self.volume = volume
		self.stock_2_trimmed = stock_2_trimmed
		self.item = item

	def check_liquidity(self, x):

		if self.avg_volume[x-1] < self.k['liquidity_min_avg_volume']:
			return False
		daily_traded_cap = self.avg_volume[x-1] * self.closes[x]
#		if daily_traded_cap < 0:
#			print "\t\tDaily Traded Cap", daily_traded_cap, avg_volume, stock_2_close[x], stock_2_trimmed[x]['Symbol'], stock_2_trimmed[x]['Date']
		if daily_traded_cap < self.k['liquidity_min_avg_cap']:
			return False
		return True

	def check_volatility(self, x):

		sigma = self.sigma_closes[x-1]
		# cancel entry if there is low volatility...
		if (sigma / self.closes[x]) < self.k['volatility_min_required'] or (sigma / self.closes[x]) > self.k['volatility_max_allowed']:
			return False
		return True


class SignalsSigmaSpan(SignalsBase):
	"""
	This version of the indicator has been shown to have a high sharpe and higher returns than the pre-existing RSI
	system. Additionally, this version allows us to exclude volumes below 100k / day, so the stocks that this trades
	are much more liquid and the simulation itself is a much more realistic picture of the market that we are trading.
	"""
	
	def __init__(self, closes, volume, stock_2_trimmed, item):

		super(SignalsSigmaSpan, self).__init__(closes, volume, stock_2_trimmed, item)

		self.k = {'sma_length': 200,
				  'sigma_closes_length': 100,
				  'avg_volume_length': 30,
				  'sigma_span_length': 5,
				  
				  'entry_sigma_span': 1.6,
				  'stop_loss_sigma_loss': 0.3,
				  'stop_loss_abs_pct_loss': 0.20,
				  'target_sigma_span': 0.8,

				  'liquidity_min_avg_volume': 100000,
				  'liquidity_min_avg_cap': 1000000,
				  'volatility_min_required': 0.060,
				  'volatility_max_allowed': 100
				 }

		self.initialize_indicators()


	def initialize_indicators(self):
		self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])

		self.sigma_closes = tools.sigma_prices(self.closes, self.k['sigma_closes_length'])
		self.avg_volume = tools.simple_moving_average(self.volume, self.k['avg_volume_length'])

		self.sigma_span = tools.sigma_span(self.closes, self.k['sigma_span_length'], sigma_input = self.sigma_closes, tr = None)

	def get_entry_signal(self, x):

		### Order really matters here!! In order for this to behave like the existing system we want to skip over all
		# the effort if the most obvious things prevent an entry. So we try to put the most common, simplest items
		# first, so we don't waste effort calculating the others unless we need to

		if not self.check_liquidity(x):
			return False

		if not self.check_volatility(x):
			return False

		sma_0 = self.sma[x]
		p_0 = self.closes[x]

		if p_0 > sma_0:
			if (self.sigma_span[x-1] > -self.k['entry_sigma_span'] and self.sigma_span[x] < -self.k['entry_sigma_span']):
				trade_result = self.get_entry_trade_result(x)
				trade_result.long_short = 'long'
				return trade_result

		elif p_0 < sma_0:
			if (self.sigma_span[x-1] < self.k['entry_sigma_span'] and self.sigma_span[x] > self.k['entry_sigma_span']):
				trade_result = self.get_entry_trade_result(x)
				trade_result.long_short = 'short'
				return trade_result

		return False

	def get_entry_trade_result(self, x):
		result = trade_result()

		result.stock_2 = self.item['stock_2']
		result.start_index = x
		result.entry_date = self.stock_2_trimmed[x]['Date']
		result.entry_price = self.closes[x]
		result.entry_vol = self.volume[x]
		result.entry_rsi = None
		result.entry_sma = self.sma[x]

		#Must use x-1 as the index here b/c this is the sigma that would have been available at entry time
		result.entry_sigma = self.sigma_closes[x-1]
		result.entry_sigma_over_p = result.entry_sigma / result.entry_price

		return result

	def get_exit(self, x, result):
		start_index = x+1
		len_data = len(self.closes)

		trading_up = True if result.long_short == 'long' else False
		trading_down = True if result.long_short == 'short' else False

		entry_sigma_over_p = result.entry_sigma_over_p
		entry_sigma = entry_sigma_over_p * result.entry_price

		# this stop loss is in terms of the # of sigma
		stop_loss = self.k['stop_loss_sigma_loss']
		pc_stop_loss = -self.k['stop_loss_abs_pct_loss']

		price_log = [self.closes[x]]

		for x in xrange(start_index, 9999999):
			if x == len_data:
				return None, None

			date_today = self.stock_2_trimmed[x]['Date']
			current_price = self.closes[x]
			price_log.append(current_price)
			price_change_pc = (current_price - result.entry_price) / result.entry_price

			if trading_up:
				ret = price_change_pc
			else:
				ret = -price_change_pc

			### print x, result.stock_2, sigma_span[x], result.entry_price, current_price
			sigma_span_diff = self.sigma_span[x] - self.sigma_span[x-1]

			if trading_up and (self.sigma_span[x] > self.k['target_sigma_span'] or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
				ret <= pc_stop_loss):

				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.exit_rsi = None
				result.end_index = x
				result.price_log = price_log

				if ret > 0:
					# print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
					result.trade_result = "Profit"
				else:
					# print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
					result.trade_result = "Loss"

				return result, result.end_index

			elif trading_down and (self.sigma_span[x] < -self.k['target_sigma_span'] or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
				ret <= pc_stop_loss):

				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.exit_rsi = None
				result.end_index = x
				result.price_log = price_log

				if ret > 0:
					# print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
					result.trade_result = "Profit"

				else:
					#print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
					result.trade_result = "Loss"

				return result, result.end_index

class SignalsSigmaSpanTest(SignalsSigmaSpan):

	def __init__(self, closes, volume, stock_2_trimmed, item):

		super(SignalsSigmaSpan, self).__init__(closes, volume, stock_2_trimmed, item)

		self.k = {'sma_length': 200,
				  'sigma_closes_length': 100,
				  'avg_volume_length': 30,
				  'sigma_span_length': 5,

				  'entry_sigma_span': 1.6,
				  'stop_loss_sigma_loss': 0.3,
				  'stop_loss_abs_pct_loss': 0.20,
				  'target_sigma_span': 0.8,

				  'liquidity_min_avg_volume': 100000,
				  'liquidity_min_avg_cap': -1000000,
				  'volatility_min_required': 0.060,
				  'volatility_max_allowed': 100
				 }

		self.initialize_indicators()

class SignalsRSISystem(SignalsBase):

	def __init__(self, closes, volume, stock_2_trimmed, item):

		super(SignalsRSISystem, self).__init__(closes, volume, stock_2_trimmed, item)

		self.k = {'sma_length': 200,
				  'sigma_closes_length': 100,
				  'avg_volume_length': 30,
				  'rsi_length': 4,

				  'entry_rsi_lower_bound': 25,
				  'entry_rsi_upper_bound': 75,
				  'stop_loss_sigma_loss': 1.4,
				  'stop_loss_abs_pct_loss': 0.06,
				  'target_rsi_long': 55,
				  'target_rsi_short': 45,
#				  'target_sigma_span': 0.8,

				  'liquidity_min_avg_volume': 0,
				  'liquidity_min_avg_cap': -1000000,
				  'volatility_min_required': 0.105,
				  'volatility_max_allowed': 100
				 }

		self.initialize_indicators()

	def initialize_indicators(self):
		self.rsi = tools.rsi(self.closes, 4)
#		self.macd_line, self.macd_signal_line  = tools.macd(self.closes, 12, 26, 9)
		self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])

		self.sigma_closes = tools.sigma_prices(self.closes, self.k['sigma_closes_length'])
		self.avg_volume = tools.simple_moving_average(self.volume, self.k['avg_volume_length'])

#		self.sigma_span = tools.sigma_span(self.closes, self.k['sigma_span_length'], sigma_input = self.sigma_closes, tr = None)

	def get_entry_signal(self, x):

		### Order really matters here!! In order for this to behave like the existing system we want to skip over all
		# the effort if the most obvious things prevent an entry. So we try to put the most common, simplest items
		# first, so we don't waste effort calculating the others unless we need to

		if not self.check_liquidity(x):
			return False

		if not self.check_volatility(x):
			return False

		sma_0 = self.sma[x]
		p_0 = self.closes[x]

		if p_0 > sma_0:
			if (self.rsi[x-1] > self.k['entry_rsi_lower_bound'] and self.rsi[x] < self.k['entry_rsi_lower_bound']):
				trade_result = self.get_entry_trade_result(x)
				trade_result.long_short = 'long'
				return trade_result

		elif p_0 < sma_0:
			if (self.rsi[x-1] < self.k['entry_rsi_upper_bound'] and self.rsi[x] > self.k['entry_rsi_upper_bound']):
				trade_result = self.get_entry_trade_result(x)
				trade_result.long_short = 'short'
				return trade_result

		return False

	def get_entry_trade_result(self, x):
		result = trade_result()

		result.stock_2 = self.item['stock_2']
		result.start_index = x
		result.entry_date = self.stock_2_trimmed[x]['Date']
		result.entry_price = self.closes[x]
		result.entry_vol = self.volume[x]
		result.entry_rsi = self.rsi[x]
		result.entry_sma = self.sma[x]

		#Must use x-1 as the index here b/c this is the sigma that would have been available at entry time
		result.entry_sigma = self.sigma_closes[x-1]
		result.entry_sigma_over_p = result.entry_sigma / result.entry_price

		return result

	def get_exit(self, x, result):
		start_index = x+1
		len_data = len(self.closes)

		trading_up = True if result.long_short == 'long' else False
		trading_down = True if result.long_short == 'short' else False

		entry_sigma_over_p = result.entry_sigma_over_p
		entry_sigma = entry_sigma_over_p * result.entry_price

		# this stop loss is in terms of the # of sigma
		stop_loss = self.k['stop_loss_sigma_loss']
		pc_stop_loss = -self.k['stop_loss_abs_pct_loss']

		price_log = [self.closes[x]]

		for x in xrange(start_index, 9999999):
			if x == len_data:
				return None, None

			date_today = self.stock_2_trimmed[x]['Date']
			current_price = self.closes[x]
			price_log.append(current_price)
			price_change_pc = (current_price - result.entry_price) / result.entry_price

			if trading_up:
				ret = price_change_pc
			else:
				ret = -price_change_pc

			### print x, result.stock_2, sigma_span[x], result.entry_price, current_price

			if trading_up and (self.rsi[x] > self.k['target_rsi_long'] or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
				ret <= pc_stop_loss):

				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.exit_rsi = self.rsi[x]
				result.end_index = x
				result.price_log = price_log

				if ret > 0:
					# print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
					result.trade_result = "Profit"
				else:
					# print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
					result.trade_result = "Loss"

				return result, result.end_index


			elif trading_down and (self.rsi[x] < self.k['target_rsi_short'] or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
				ret <= pc_stop_loss):

				result.time_in_trade = x - (start_index - 1)
				result.exit_price = current_price
				result.ret = ret
				result.chained_ret = 1 + ret
				result.exit_date = date_today
				result.exit_rsi = self.rsi[x]
				result.end_index = x
				result.price_log = price_log

				if ret > 0:
					# print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
					result.trade_result = "Profit"

				else:
					#print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
					result.trade_result = "Loss"

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
		self.entry_vol = 0
		self.entry_rsi = 0
		self.exit_rsi = 0
		self.entry_sma = 0
		self.entry_sigma = 0
		self.entry_sigma_over_p = 0

		self.price_log = []

		self.ret = 0
		self.chained_ret = 0
		self.time_in_trade = 0
		self.trade_result = '' # profit, loss, timeout

