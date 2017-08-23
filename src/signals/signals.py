
import src.toolsx as tools
import src.math_tools as math_tools
import math
import numpy as np
import datetime

from scipy import stats
import statsmodels.tsa.stattools as statsmodels

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

class InWeekAfterDown(SignalsBase):

    def __init__(self, closes, volume, stock_2_trimmed, item):
        super(InWeekAfterDown, self).__init__(closes, volume, stock_2_trimmed, item)

        self.get_weekly_closes()

    def get_weekly_closes(self):
        """Finds starts/ends of weeks and gets the total return across that week. If end of week is detected, simply
        add a key to stock_2_trimmed"""

        for x, day_dict in enumerate(self.stock_2_trimmed):
            if x <= 4:
                continue
            today_int = day_dict['Date']
            today_datetime = datetime.datetime(int(today_int[0:4]), int(today_int[4:6]), int(today_int[6:8]))
            # Monday is 0, Sunday is 7
            today_day_of_week = today_datetime.weekday()

            yesterday_int = self.stock_2_trimmed[x-1]['Date']
            yesterday_datetime = datetime.datetime(int(yesterday_int[0:4]), int(yesterday_int[4:6]), int(yesterday_int[6:8]))
            # Monday is 0, Sunday is 7
            yesterday_day_of_week = yesterday_datetime.weekday()

            # print "Today, today weekday, yesterday, yesterday weekday: ", today_int, today_day_of_week, yesterday_int, yesterday_day_of_week

            if today_day_of_week < yesterday_day_of_week:
                # print "starting iteration: ", today_day_of_week, yesterday_day_of_week

                prev_iter_weekday = yesterday_day_of_week
                x_starting = x-1
                while True:
                    x_starting -= 1
                    # print "x starting: ", x_starting
                    current_weekday_int = self.stock_2_trimmed[x_starting]['Date']
                    current_weekday_datetime = datetime.datetime(int(current_weekday_int[0:4]), int(current_weekday_int[4:6]), int(current_weekday_int[6:8]))
                    iter_weekday = current_weekday_datetime.weekday()
                    if iter_weekday >= prev_iter_weekday:
                        break
                    else:
                        prev_iter_weekday = iter_weekday

                # This line is correct and should be used
                # print "Week start/end: ", self.stock_2_trimmed[x_starting+1]['Date'], prev_iter_weekday, yesterday_int, yesterday_day_of_week

                # x starting here is actually the day before the week starts, this gives us the previous Friday's close
                last_week_open = self.stock_2_trimmed[x_starting]['AdjClose']
                last_week_close = self.stock_2_trimmed[x-1]['AdjClose']
                last_week_return = last_week_close / last_week_open
                self.stock_2_trimmed[x-1]['week_return'] = last_week_return
                print "Week Return: ", self.stock_2_trimmed[x_starting+1]['Date'], yesterday_int, last_week_return

    def get_entry_signal(self, x):
        week_return = self.stock_2_trimmed[x].get('week_return')
        if week_return:
            if week_return > 0.97 and week_return < 0.995:
                trade_result = self.get_entry_trade_result(x)
                trade_result.long_short = 'long'
                return trade_result
        else:
            return False

    def get_entry_trade_result(self, x):
        result = trade_result()

        result.stock_2 = self.item['stock_2']
        result.entry_date = self.stock_2_trimmed[x]['Date']
        result.entry_price = self.closes[x]
        result.entry_vol = self.volume[x]
        result.prev_weekly_close = self.stock_2_trimmed[x]['week_return']

        return result

    def get_exit(self, x, result):
        start_index = x+1
        len_data = len(self.closes)

        trading_up = True if result.long_short == 'long' else False

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

            if trading_up and self.stock_2_trimmed[x].get('week_return'):

                result.time_in_trade = x - (start_index - 1)
                result.exit_price = current_price
                result.ret = ret
                result.chained_ret = 1 + ret
                result.exit_date = date_today
                result.end_index = x
                result.price_log = price_log

                if ret > 0:
                    # print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Profit"
                else:
                    # print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Loss"

                return result, result.end_index

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

        self.k = {'sma_length': 175,
                  'sigma_closes_length': 100,
                  'avg_volume_length': 30,
                  'sigma_span_length': 5,

                  'entry_sigma_span': 1.6,
                  'stop_loss_sigma_loss': 0.3,
                  'stop_loss_abs_pct_loss': 0.06,
                  'target_sigma_span': 0.8,

                  'liquidity_min_avg_volume': 100000,
                  'liquidity_min_avg_cap': 2500000,
                  'volatility_min_required': 0.060,
                  'volatility_max_allowed': 100
                 }

        self.initialize_indicators()

    def initialize_indicators(self):
        self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])
        self.sma_2 = tools.simple_moving_average(self.closes, 50)

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


        # Only enter into markets where we are with the long term trend AND the trend is actually trending (not sideways)
        sma_diff_pct = abs((self.sma_2[x] - self.sma_2[x-1]) / self.sma_2[x-1])
        sma_annualized_pct = sma_diff_pct * 252

#		print "diff pct, ann", sma_diff_pct, sma_annualized_pct

        if sma_annualized_pct < 0.00:
            return False



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

class SignalsSigmaSpanVolatilityTest_3(SignalsSigmaSpan):
    """
    Similar to the original SignalsSigmaSpanVolatilityTest, but this one does not check volatility based on sigma/p.
    Instead this looks for points where the short term volatility is higher than the longer term volatility, which
    suggests rising and more recent volatility.
    """

    def __init__(self, closes, volume, stock_2_trimmed, item, is_stock = False, kwargs=None):

        super(SignalsSigmaSpan, self).__init__(closes, volume, stock_2_trimmed, item)

        self.k = {'sma_length': 175,
                  'sigma_closes_length': 100,
                  'avg_volume_length': 30,

                  'entry_sigma_span': 1.6,
                  'stop_loss_sigma_loss': 2.0,
                  'stop_loss_abs_pct_loss': 0.06,

                  # target sigma span is an exit based on the actual sigma span value at that particular day
                  'target_sigma_span': 100,     # setting to a very high positive number effectively disables this
                  'target_volatility_multiple': 100, # 1.6,


                  # the sigma span target sigma multiple looks at the return in the trade and exits if it is this
                  # multiple of the historical sigma span's sigma that was used
                  'sigma_span_target_sigma_multiple': 0.1, # set to high positive # to disable
                  'sigma_span_length': 5,
                  'sigma_span_historical_lookback': 100,
                  'exit_days': 4,

                  'liquidity_min_avg_volume': 100000,
                  'liquidity_min_avg_cap': 2500000,
                  'volatility_min_required': 0.040,
                  'short_volatility_percentile': 80,

                  'volatility_max_allowed': 100,
                  'volatility_long_lookback': 100
                 }

        if is_stock:
            self.k['entry_sigma_span'] = 1.9
            ### this was a fortuitous error: setting abs stop loss to 1.8 (1800%) instead of 0.06 and then NOT changing target vol_multiple
            # This effectively disabled the stoploss altogether...
            self.k['stop_loss_abs_pct_loss'] = 0.08
            self.k['target_volatility_multiple'] = 100 # 1.9

            self.k['volatility_min_required'] = 0.060
            self.k['short_volatility_percentile'] = 85

        if kwargs:
            self.stock_1_closes = kwargs['stock_1_close']
            self.stock_1_sma = tools.memoized_simple_moving_average(self.stock_1_closes, 200)

        self.initialize_indicators()

    def check_volatility(self, x):

        vol_short = self.volatility[x-1]
        vol_long = self.volatility_long[x-1]
        # we expect to see a period of shorter term volatility that has recently started, or increased
        vol_diff_0 = vol_short - vol_long
        vol_diff_2 = (self.volatility[x-1] - self.volatility_long[x-1]) - (self.volatility[x-3] - self.volatility_long[x-3])
        if vol_short < vol_long:
            return False

        if self.volatility[x] < self.ref_vol:
            return False

#        macd_vol = self.macd_line_volatility[x-1]
#        signal_vol = self.signal_line_volatility[x-1]
#        histogram_vol = macd_vol - signal_vol
#
#        if histogram_vol <= 0:
#            return False

        sigma = self.sigma_closes[x-1]
        # cancel entry if there is low volatility...
        if (sigma / self.closes[x]) < self.k['volatility_min_required'] or (sigma / self.closes[x]) > self.k['volatility_max_allowed']:
            return False
        return True

        return True

    def initialize_indicators(self):
        self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])
        self.avg_volume = tools.simple_moving_average(self.volume, self.k['avg_volume_length'])

#        self.short_sma = tools.simple_moving_average(self.closes, 3)
#        temp = [0,0]
#        temp.extend(self.short_sma)
#        self.short_sma = temp # this effectively shifts it by 3

        # sigma_closes is convenient because it is in terms of dollars and can be easily used to set a dollar-based
        # stop loss. It is correlated with the volatility, but they are not scaled, so it is worth testing a stop loss
        # based on the 100 day volatility...
        self.sigma_closes = tools.sigma_prices(self.closes, self.k['sigma_closes_length'])

        self.volatility = tools.volatility_bs_annualized(self.closes, 30, returns_period_length=self.k['sigma_span_length'])


        self.sma_20 = tools.simple_moving_average(self.closes, 20)
        temp = [0, 0, 0]
        temp.extend(tools.simple_moving_average(self.closes, 3))
        self.sma_33 = temp


        # self.macd_line_volatility, self.signal_line_volatility = tools.macd(self.volatility, 12, 26, 9)

        # when we use a long lookback, sometimes we don't have enough data, so only lookback as far as we have data
        volatility_long_lookback = min(self.k['volatility_long_lookback'], (len(self.closes) - 10))
        self.volatility_long = tools.volatility_bs_annualized(self.closes, volatility_long_lookback, returns_period_length=self.k['sigma_span_length'])

        self.ref_vol = stats.scoreatpercentile(self.volatility[-1008:], self.k['short_volatility_percentile'])

        # self.returns = math_tools.get_returns(self.closes)

        # self.kurtosis = np.empty(len(self.returns))
#        for k, ret in enumerate(self.returns):
#            self.kurtosis[k] = stats.kurtosis(self.returns[k-504:], fisher=False, bias=True)

        self.sigma_span, self.historical_sigma = tools.sigma_span(self.closes, self.k['sigma_span_length'], self.k['sigma_span_historical_lookback'])

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

        target_factor = np.sqrt(252/self.k['sigma_span_length'])

        if p_0 > sma_0:
            if (self.sigma_span[x-1] > -self.k['entry_sigma_span'] and self.sigma_span[x] < -self.k['entry_sigma_span']):
                trade_result = self.get_entry_trade_result(x)
                trade_result.long_short = 'long'
                trade_result.target = (1 + (self.k['target_volatility_multiple'] * self.volatility[x] / target_factor)) * trade_result.entry_price
                return trade_result

#        elif p_0 < sma_0:
#            if (self.sigma_span[x-1] < self.k['entry_sigma_span'] and self.sigma_span[x] > self.k['entry_sigma_span']):
#                trade_result = self.get_entry_trade_result(x)
#                trade_result.long_short = 'short'
#                trade_result.target = (1 - (self.k['target_volatility_multiple'] * self.volatility[x] / target_factor)) * trade_result.entry_price
#                return trade_result

        return False

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
        date_log = [self.stock_2_trimmed[x]['Date']]

        for x in xrange(start_index, 9999999):
            if x == len_data:
                return None, None

            date_today = self.stock_2_trimmed[x]['Date']
            current_price = self.closes[x]
            price_log.append(current_price)
            date_log.append(date_today)
            price_change_pc = (current_price - result.entry_price) / result.entry_price

            if trading_up:
                ret = price_change_pc
            else:
                ret = -price_change_pc

            time_in = x - (start_index - 1)
            exit_time = self.k['exit_days']
            exit_after_loss = 999

            ### print x, result.stock_2, sigma_span[x], result.entry_price, current_price
            sigma_span_diff = self.sigma_span[x] - self.sigma_span[x-1]

#            if trading_up and (self.sigma_span[x] > self.k['target_sigma_span'] or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):


            # (time_in >= exit_time and current_price < self.short_sma[x]) or \
#            if trading_up and ( \
#                time_in == exit_time or \
#                current_price > result.target or \
#                current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss or \
#                (ret < 0 and time_in > exit_after_loss) or \
#                self.sigma_span[x] > self.k['target_sigma_span'] or \
#                ret > self.k['sigma_span_target_sigma_multiple']):

            if trading_up and ( \
                (time_in == 4 and ret < 0) or \
                (time_in > 4 and time_in <= 12 and current_price < self.sma_33[x]) or \
                (time_in > 12 and current_price < self.sma_20[x]) or \
                ret <= pc_stop_loss or \
                ret > (self.k['sigma_span_target_sigma_multiple'] * self.volatility[x])
            ):
#                current_price < (result.entry_price - (stop_loss * entry_sigma)) or \

                if ret > (self.k['sigma_span_target_sigma_multiple'] * self.volatility[x]):
                    print "****", ret, self.volatility[x], (self.k['sigma_span_target_sigma_multiple'] * self.volatility[x])
#
#               if trading_up and (self.closes[x] > result.target or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

                result.time_in_trade = x - (start_index - 1)
                result.exit_price = current_price
                result.ret = ret
                result.chained_ret = 1 + ret
                result.exit_date = date_today
                result.exit_rsi = None
                result.end_index = x
                result.price_log = price_log
                result.date_log = date_log

                if ret > 0:
                    # print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Profit"
                else:
                    # print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Loss"

                return result, result.end_index

#            elif trading_down and (self.sigma_span[x] < -self.k['target_sigma_span'] or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

            elif trading_down and (time_in == exit_time or \
                current_price < result.target or \
                current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
                ret <= pc_stop_loss or \
                (ret < 0 and time_in > exit_after_loss) or \
                self.sigma_span[x] < -self.k['target_sigma_span'] or \
                ret > self.k['sigma_span_target_sigma_multiple']):

#            elif trading_down and (self.closes[x] < result.target or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

                result.time_in_trade = x - (start_index - 1)
                result.exit_price = current_price
                result.ret = ret
                result.chained_ret = 1 + ret
                result.exit_date = date_today
                result.exit_rsi = None
                result.end_index = x
                result.price_log = price_log
                result.date_log = date_log

                if ret > 0:
                    # print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Profit"

                else:
                    #print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Loss"

                return result, result.end_index

    def get_entry_trade_result(self, x):

        # print 'Kurtosis: ', x, self.kurtosis[x], '\t', self.volatility[x]

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

        # result.entry_score = np.mean(self.volatility_long[-1008:])
        # result.entry_score = abs(self.volatility[x-1] - self.volatility_long[x-1])
        result.entry_score = self.volatility_long[x]
        # result.entry_score = result.entry_sigma_over_p

        result.entry_sigma_percentile = stats.percentileofscore(self.volatility, self.volatility[x])
        result.entry_volatility = self.volatility[x]

        return result

class SignalsSigmaSpanVolatilityTest_2(SignalsSigmaSpan):
    """
    Similar to the original SignalsSigmaSpanVolatilityTest, but this one does not check volatility based on sigma/p.
    Instead this looks for points where the short term volatility is higher than the longer term volatility, which
    suggests rising and more recent volatility.
    """

    def __init__(self, closes, volume, stock_2_trimmed, item, is_stock = False):

        super(SignalsSigmaSpan, self).__init__(closes, volume, stock_2_trimmed, item)

        self.k = {'sma_length': 175,
                  'sigma_closes_length': 100,
                  'avg_volume_length': 30,

                  'entry_sigma_span': 1.6,
                  'stop_loss_sigma_loss': 2.0,
                  'stop_loss_abs_pct_loss': 0.06,

                  # target sigma span is an exit based on the actual sigma span value at that particular day
                  'target_sigma_span': 100,     # setting to a very high positive number effectively disables this
                  'target_volatility_multiple': 100, # 1.6,


                  # the sigma span target sigma multiple looks at the return in the trade and exits if it is this
                  # multiple of the historical sigma span's sigma that was used
                  'sigma_span_target_sigma_multiple': 0.05, # set to high positive # to disable
                  'sigma_span_length': 5,
                  'sigma_span_historical_lookback': 100,
                  'exit_days': 4,

                  'liquidity_min_avg_volume': 100000,
                  'liquidity_min_avg_cap': 2500000,
                  'volatility_min_required': 0.040,
                  'short_volatility_percentile': 80,

                  'volatility_max_allowed': 100,
                  'volatility_long_lookback': 100
                 }

        if is_stock:
            self.k['entry_sigma_span'] = 1.9
            ### this was a fortuitous error: setting abs stop loss to 1.8 (1800%) instead of 0.06 and then NOT changing target vol_multiple
            # This effectively disabled the stoploss altogether...
            self.k['stop_loss_abs_pct_loss'] = 0.08
            self.k['target_volatility_multiple'] = 100 # 1.9

            self.k['volatility_min_required'] = 0.060
            self.k['short_volatility_percentile'] = 85

        self.initialize_indicators()

    def check_volatility(self, x):

        vol_short = self.volatility[x-1]
        vol_long = self.volatility_long[x-1]
        # we expect to see a period of shorter term volatility that has recently started, or increased
        vol_diff_0 = vol_short - vol_long
        vol_diff_2 = (self.volatility[x-1] - self.volatility_long[x-1]) - (self.volatility[x-3] - self.volatility_long[x-3])
        if vol_short < vol_long:
            return False

        if self.volatility[x] < self.ref_vol:
            return False

#        macd_vol = self.macd_line_volatility[x-1]
#        signal_vol = self.signal_line_volatility[x-1]
#        histogram_vol = macd_vol - signal_vol
#
#        if histogram_vol <= 0:
#            return False

        sigma = self.sigma_closes[x-1]
        # cancel entry if there is low volatility...
        if (sigma / self.closes[x]) < self.k['volatility_min_required'] or (sigma / self.closes[x]) > self.k['volatility_max_allowed']:
            return False
        return True

        return True

    def initialize_indicators(self):
        self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])
        self.avg_volume = tools.simple_moving_average(self.volume, self.k['avg_volume_length'])

#        self.short_sma = tools.simple_moving_average(self.closes, 3)
#        temp = [0,0]
#        temp.extend(self.short_sma)
#        self.short_sma = temp # this effectively shifts it by 3

        # sigma_closes is convenient because it is in terms of dollars and can be easily used to set a dollar-based
        # stop loss. It is correlated with the volatility, but they are not scaled, so it is worth testing a stop loss
        # based on the 100 day volatility...
        self.sigma_closes = tools.sigma_prices(self.closes, self.k['sigma_closes_length'])

        self.volatility = tools.volatility_bs_annualized(self.closes, 30, returns_period_length=self.k['sigma_span_length'])

        # self.macd_line_volatility, self.signal_line_volatility = tools.macd(self.volatility, 12, 26, 9)

        # when we use a long lookback, sometimes we don't have enough data, so only lookback as far as we have data
        volatility_long_lookback = min(self.k['volatility_long_lookback'], (len(self.closes) - 10))
        self.volatility_long = tools.volatility_bs_annualized(self.closes, volatility_long_lookback, returns_period_length=self.k['sigma_span_length'])

        self.ref_vol = stats.scoreatpercentile(self.volatility[-1008:], self.k['short_volatility_percentile'])

        # self.returns = math_tools.get_returns(self.closes)

        # self.kurtosis = np.empty(len(self.returns))
#        for k, ret in enumerate(self.returns):
#            self.kurtosis[k] = stats.kurtosis(self.returns[k-504:], fisher=False, bias=True)

        self.sigma_span, self.historical_sigma = tools.sigma_span(self.closes, self.k['sigma_span_length'], self.k['sigma_span_historical_lookback'])

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

        target_factor = np.sqrt(252/self.k['sigma_span_length'])

        if p_0 > sma_0:
            if (self.sigma_span[x-1] > -self.k['entry_sigma_span'] and self.sigma_span[x] < -self.k['entry_sigma_span']):
                trade_result = self.get_entry_trade_result(x)
                trade_result.long_short = 'long'
                trade_result.target = (1 + (self.k['target_volatility_multiple'] * self.volatility[x] / target_factor)) * trade_result.entry_price
                return trade_result

#        elif p_0 < sma_0:
#            if (self.sigma_span[x-1] < self.k['entry_sigma_span'] and self.sigma_span[x] > self.k['entry_sigma_span']):
#                trade_result = self.get_entry_trade_result(x)
#                trade_result.long_short = 'short'
#                trade_result.target = (1 - (self.k['target_volatility_multiple'] * self.volatility[x] / target_factor)) * trade_result.entry_price
#                return trade_result

        return False

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
        date_log = [self.stock_2_trimmed[x]['Date']]

        for x in xrange(start_index, 9999999):
            if x == len_data:
                return None, None

            date_today = self.stock_2_trimmed[x]['Date']
            current_price = self.closes[x]
            price_log.append(current_price)
            date_log.append(date_today)
            price_change_pc = (current_price - result.entry_price) / result.entry_price

            if trading_up:
                ret = price_change_pc
            else:
                ret = -price_change_pc

            time_in = x - (start_index - 1)
            exit_time = self.k['exit_days']
            exit_after_loss = 999

            ### print x, result.stock_2, sigma_span[x], result.entry_price, current_price
            sigma_span_diff = self.sigma_span[x] - self.sigma_span[x-1]

#            if trading_up and (self.sigma_span[x] > self.k['target_sigma_span'] or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):


            # (time_in >= exit_time and current_price < self.short_sma[x]) or \
            if trading_up and ( \
                time_in == exit_time or \
                current_price > result.target or \
                current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
                ret <= pc_stop_loss or \
                (ret < 0 and time_in > exit_after_loss) or \
                self.sigma_span[x] > self.k['target_sigma_span'] or \
                ret > self.k['sigma_span_target_sigma_multiple']):

#            if trading_up and (self.closes[x] > result.target or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

                result.time_in_trade = x - (start_index - 1)
                result.exit_price = current_price
                result.ret = ret
                result.chained_ret = 1 + ret
                result.exit_date = date_today
                result.exit_rsi = None
                result.end_index = x
                result.price_log = price_log
                result.date_log = date_log

                if ret > 0:
                    # print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Profit"
                else:
                    # print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Loss"

                return result, result.end_index

#            elif trading_down and (self.sigma_span[x] < -self.k['target_sigma_span'] or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

            elif trading_down and (time_in == exit_time or \
                current_price < result.target or \
                current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
                ret <= pc_stop_loss or \
                (ret < 0 and time_in > exit_after_loss) or \
                self.sigma_span[x] < -self.k['target_sigma_span'] or \
                ret > self.k['sigma_span_target_sigma_multiple']):

#            elif trading_down and (self.closes[x] < result.target or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

                result.time_in_trade = x - (start_index - 1)
                result.exit_price = current_price
                result.ret = ret
                result.chained_ret = 1 + ret
                result.exit_date = date_today
                result.exit_rsi = None
                result.end_index = x
                result.price_log = price_log
                result.date_log = date_log

                if ret > 0:
                    # print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Profit"

                else:
                    #print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Loss"

                return result, result.end_index

    def get_entry_trade_result(self, x):

        # print 'Kurtosis: ', x, self.kurtosis[x], '\t', self.volatility[x]

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

        # result.entry_score = np.mean(self.volatility_long[-1008:])
        # result.entry_score = abs(self.volatility[x-1] - self.volatility_long[x-1])
        result.entry_score = self.volatility_long[x]
        # result.entry_score = result.entry_sigma_over_p

        result.entry_sigma_percentile = stats.percentileofscore(self.volatility, self.volatility[x])
        result.entry_volatility = self.volatility[x]

        return result

class SignalsSigmaSpanVolatilityTest(SignalsSigmaSpan):

    def __init__(self, closes, volume, stock_2_trimmed, item):

        super(SignalsSigmaSpan, self).__init__(closes, volume, stock_2_trimmed, item)

        self.k = {'sma_length': 175,
                  'sigma_closes_length': 100,
                  'avg_volume_length': 30,
                  'sigma_span_length': 5,
                  'sigma_span_historical_lookback': 100,

                  'entry_sigma_span': 1.6,
                  'stop_loss_sigma_loss': 2.0,
                  'stop_loss_abs_pct_loss': 0.06,
                  'target_sigma_span': 0.8,

                  'liquidity_min_avg_volume': 100000,
                  'liquidity_min_avg_cap': 2500000,
                  # 'volatility_min_required': 0.25,
                  'volatility_min_required': 0.040,
                  'volatility_max_allowed': 100
                 }

        self.initialize_indicators()

    def initialize_indicators(self):
        self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])

        self.volatility = tools.volatility_bs_annualized(self.closes, 30, returns_period_length=5)
        self.ref_vol = stats.scoreatpercentile(self.volatility[-1008:], 80)

        self.sigma_closes = tools.sigma_prices(self.closes, self.k['sigma_closes_length'])
        self.avg_volume = tools.simple_moving_average(self.volume, self.k['avg_volume_length'])

        self.sigma_span, self.historical_sigma = tools.sigma_span(self.closes, self.k['sigma_span_length'], self.k['sigma_span_historical_lookback'])

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

        ref_vol = 1.0 * self.ref_vol

        if self.volatility[x] < ref_vol:
            return False

        if p_0 > sma_0:
            if (self.sigma_span[x-1] > -self.k['entry_sigma_span'] and self.sigma_span[x] < -self.k['entry_sigma_span']):
                trade_result = self.get_entry_trade_result(x)
                trade_result.long_short = 'long'
                trade_result.target = 1.2 * (1 + self.historical_sigma[x]) * trade_result.entry_price
                return trade_result

        elif p_0 < sma_0:
            if (self.sigma_span[x-1] < self.k['entry_sigma_span'] and self.sigma_span[x] > self.k['entry_sigma_span']):
                trade_result = self.get_entry_trade_result(x)
                trade_result.long_short = 'short'
                trade_result.target = 1.2 * (1 - self.historical_sigma[x]) * trade_result.entry_price
                return trade_result

        return False

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

            time_in = x - (start_index - 1)
            exit_time = 4
            exit_after_loss = 999

            ### print x, result.stock_2, sigma_span[x], result.entry_price, current_price
            sigma_span_diff = self.sigma_span[x] - self.sigma_span[x-1]

#            if trading_up and (self.sigma_span[x] > self.k['target_sigma_span'] or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

            if trading_up and (time_in == exit_time or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
                ret <= pc_stop_loss or (ret < 0 and time_in > exit_after_loss)):

#            if trading_up and (self.closes[x] > result.target or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

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

#            elif trading_down and (self.sigma_span[x] < -self.k['target_sigma_span'] or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

            elif trading_down and (time_in == exit_time or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
                ret <= pc_stop_loss or (ret < 0 and time_in > exit_after_loss)):

#            elif trading_down and (self.closes[x] < result.target or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
#                ret <= pc_stop_loss):

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

        result.entry_score = result.entry_sigma_over_p

        result.entry_sigma_percentile = stats.percentileofscore(self.volatility, self.volatility[x])
        result.entry_volatility = self.volatility[x]

        return result

class MomentumVolatilityTest(SignalsBase):

    def __init__(self, closes, volume, stock_2_trimmed, item):

        super(MomentumVolatilityTest, self).__init__(closes, volume, stock_2_trimmed, item)

        self.k = {'sma_length': 200,
                  'sigma_closes_length': 100,
                  'avg_volume_length': 30,
                  # 'sigma_span_length': 5,
                  # 'sigma_span_historical_lookback': 100,

                  # 'entry_sigma_span': 1.6,
                  'stop_loss_sigma_loss': 1.0,
                  'stop_loss_abs_pct_loss': 0.05,
                  # 'target_sigma_span': 0.8,

                  'liquidity_min_avg_volume': 100000,
                  'liquidity_min_avg_cap': 2500000,
                  'volatility_min_required': 0.060,
                  'volatility_max_allowed': 100
                 }

        self.initialize_indicators()

    def initialize_indicators(self):
        self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])

        self.sma_2 = tools.simple_moving_average(self.closes, 2)
        s3 = tools.simple_moving_average(self.closes, 3)
        s3_shifted = np.empty(len(s3))
        for x in range(3, len(s3)):
            s3_shifted[x] = s3[x-3]
        self.sma_3 = s3_shifted

        self.volatility = tools.volatility_bs_annualized(self.closes, 30, returns_period_length=5)
        self.mean_vol = stats.scoreatpercentile(self.volatility[-1008:], 40)

        self.sigma_closes = tools.sigma_prices(self.closes, self.k['sigma_closes_length'])
        self.avg_volume = tools.simple_moving_average(self.volume, self.k['avg_volume_length'])

        self.macd_line, self.macd_signal_line  = tools.macd(self.closes, 12, 26, 9)
        # self.sigma_span = tools.sigma_span(self.closes, self.k['sigma_span_length'], self.k['sigma_span_historical_lookback'], sigma_input = self.sigma_closes)

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

        ref_vol = 1.0 * self.mean_vol

        if self.volatility[x] > ref_vol:
            return False

        p0_macd_histogram = self.macd_line[x] - self.macd_signal_line[x]
        p1_macd_histogram = self.macd_line[x-1] - self.macd_signal_line[x-1]

        crossing_up = p0_macd_histogram > 0 and p1_macd_histogram < 0
        crossing_down = p0_macd_histogram < 0 and p1_macd_histogram > 0

        xmas_up = self.sma_2[x] > self.sma_3[x] and self.sma_2[x-1] < self.sma_3[x-1]
        xmas_down = self.sma_2[x] > self.sma_3[x] and self.sma_2[x-1] < self.sma_3[x-1]


        if p_0 > sma_0:
#            if (self.sigma_span[x-1] > -self.k['entry_sigma_span'] and self.sigma_span[x] < -self.k['entry_sigma_span']):
            if crossing_up:
                trade_result = self.get_entry_trade_result(x)
                trade_result.long_short = 'long'
                return trade_result

        elif p_0 < sma_0:
#            if (self.sigma_span[x-1] < self.k['entry_sigma_span'] and self.sigma_span[x] > self.k['entry_sigma_span']):
            if crossing_down:
                trade_result = self.get_entry_trade_result(x)
                trade_result.long_short = 'short'
                return trade_result

        return False

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
            # sigma_span_diff = self.sigma_span[x] - self.sigma_span[x-1]

            p0_macd_histogram = self.macd_line[x] - self.macd_signal_line[x]
            p1_macd_histogram = self.macd_line[x-1] - self.macd_signal_line[x-1]

            crossing_up = p0_macd_histogram > 0 and p1_macd_histogram < 0
            crossing_down = p0_macd_histogram < 0 and p1_macd_histogram > 0

            xmas_up = self.sma_2[x] > self.sma_3[x] and self.sma_2[x-1] < self.sma_3[x-1]
            xmas_down = self.sma_2[x] > self.sma_3[x] and self.sma_2[x-1] < self.sma_3[x-1]

            if trading_up and (xmas_down or current_price < (result.entry_price - (stop_loss * entry_sigma)) or \
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

            elif trading_down and (xmas_up or current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
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

        result.entry_sigma_percentile = stats.percentileofscore(self.volatility, self.volatility[x])

        return result


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

class SignalsRSISystemTest(SignalsRSISystem):

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

                  'liquidity_min_avg_volume': 0,
                  'liquidity_min_avg_cap': -1000000,
                  'volatility_min_required': 0.105,
                  'volatility_max_allowed': 100
                 }

        self.initialize_indicators()

class SignalsRSISystemVolatilityTest(SignalsRSISystem):

    def __init__(self, closes, volume, stock_2_trimmed, item):

        super(SignalsRSISystem, self).__init__(closes, volume, stock_2_trimmed, item)

        self.k = {'sma_length': 175,
                  'sigma_closes_length': 100,
                  'avg_volume_length': 30,
                  'rsi_length': 4,

                  'entry_rsi_lower_bound': 25,
                  'entry_rsi_upper_bound': 75,
                  'stop_loss_sigma_loss': 1.4,
                  'stop_loss_abs_pct_loss': 0.06,
                  'target_rsi_long': 55,
                  'target_rsi_short': 45,

                  'liquidity_min_avg_volume': 100000,
                  'liquidity_min_avg_cap': -1,
                  'volatility_min_required': 0.08,
                  'volatility_max_allowed': 100
                 }

        self.initialize_indicators()


    def initialize_indicators(self):
        self.rsi = tools.rsi(self.closes, self.k['rsi_length'])
#		self.macd_line, self.macd_signal_line  = tools.macd(self.closes, 12, 26, 9)
        self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])

        self.volatility = tools.volatility_bs_annualized(self.closes, 30, returns_period_length=4)
        ### self.mean_vol = np.median(self.volatility)
        self.mean_vol = stats.scoreatpercentile(self.volatility, 80)

        self.sigma_closes = tools.sigma_prices(self.closes, self.k['sigma_closes_length'])
        self.avg_volume = tools.simple_moving_average(self.volume, self.k['avg_volume_length'])

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

        ref_vol = 1.0 * self.mean_vol

        if self.volatility[x] < ref_vol:
            return False

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
        result.entry_rsi = None
        result.entry_sma = self.sma[x]

        #Must use x-1 as the index here b/c this is the sigma that would have been available at entry time
        result.entry_sigma = self.sigma_closes[x-1]
        result.entry_sigma_over_p = result.entry_sigma / result.entry_price

        result.entry_sigma_percentile = stats.percentileofscore(self.volatility, self.volatility[x])

        return result




















class MovingAverageSeasonalitySystem(SignalsSigmaSpan):
    """
    Similar to the original SignalsSigmaSpanVolatilityTest, but this one does not check volatility based on sigma/p.
    Instead this looks for points where the short term volatility is higher than the longer term volatility, which
    suggests rising and more recent volatility.
    """

    def __init__(self, closes, volume, stock_2_trimmed, item, is_stock = False, kwargs = None):

        super(MovingAverageSeasonalitySystem, self).__init__(closes, volume, stock_2_trimmed, item)

        self.k = {'sma_length': 200,
#                  'sigma_closes_length': 100,
                  'avg_volume_length': 30,

                  'sigma_closes_length': 200,

                  'volatility_min_required': 0.00,
                  'volatility_max_allowed': 100,

                  'stop_loss_abs_pct_loss': 0.05,

                  'entry_month': 11,
                  'entry_day': 10,
                  'exit_month': 05,
                  'exit_day': 10,

                  'liquidity_min_avg_volume': 100000,
                  'liquidity_min_avg_cap': 2500000,
                 }

        if kwargs:
            self.stock_1_closes = kwargs['stock_1_close']
            self.stock_1_sma = tools.memoized_simple_moving_average(self.stock_1_closes, 200)

        self.initialize_indicators()


    def initialize_indicators(self):
        self.sma = tools.simple_moving_average(self.closes, self.k['sma_length'])
        self.avg_volume = tools.simple_moving_average(self.volume, self.k['avg_volume_length'])
        self.sigma_closes = tools.sigma_prices(self.closes, self.k['sigma_closes_length'])

#        self.stock_1_sma = tools.memoized_simple_moving_average(self.stock_1_closes, 200)

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

        current_date = self.stock_2_trimmed[x]['Date']
        current_date = datetime.datetime.strptime(current_date, '%Y-%m-%d')
        current_year = current_date.year
        current_year_entry_date = datetime.datetime(current_year, self.k['entry_month'], self.k['entry_day'])
        current_year_exit_date = datetime.datetime(current_year, self.k['exit_month'], self.k['exit_day'])

        score = 0

        if current_date > current_year_entry_date or current_date < current_year_exit_date:
            score += 0

        if p_0 > sma_0 and self.closes[x-1] > self.sma[x-1]:
        # if self.stock_1_closes[x-1] > self.stock_1_sma[x-1] and self.sma[x-1] > self.sma[x-2] and self.closes[x-1] > self.sma[x-1] and self.closes[x-22] < self.sma[x-22]:
        # if self.stock_1_closes[x-1] > self.stock_1_sma[x-1] and \
           # self.closes[x-1] > self.sma[x-1] and self.closes[x-2] > self.sma[x-2] and self.closes[x-3] < self.sma[x-3]:
            # score += 1
            # The weekday ordinal = 0 for M, 6 for Sun, so if tomorrow's number is less than today's then today is last
            # trading day of the current week
            try:
                # if current_date.weekday() > datetime.datetime.strptime(self.stock_2_trimmed[x+1]['Date'], '%Y-%m-%d').weekday():
                if current_date.month != datetime.datetime.strptime(self.stock_2_trimmed[x+1]['Date'], '%Y-%m-%d').month:
                    score += 1
            except:
                pass

        if score > 0:
            trade_result = self.get_entry_trade_result(x)
            trade_result.long_short = 'long'
            # trade_result.target = (1 + (self.k['target_volatility_multiple'] * self.volatility[x] / target_factor)) * trade_result.entry_price
            return trade_result

#        elif p_0 < sma_0:
#            trade_result = self.get_entry_trade_result(x)
#            trade_result.long_short = 'short'
#            trade_result.target = (1 - (self.k['target_volatility_multiple'] * self.volatility[x] / target_factor)) * trade_result.entry_price
#            return trade_result

        return False

    def get_exit(self, x, result):
        start_index = x+1
        len_data = len(self.closes)

        trading_up = True if result.long_short == 'long' else False
        trading_down = True if result.long_short == 'short' else False

        entry_sigma_over_p = result.entry_sigma_over_p
        entry_sigma = entry_sigma_over_p * result.entry_price

        # this stop loss is in terms of the # of sigma
        # stop_loss = self.k['stop_loss_sigma_loss']
        pc_stop_loss = -self.k['stop_loss_abs_pct_loss']

        price_log = [self.closes[x]]
        date_log = [self.stock_2_trimmed[x]['Date']]

        for x in xrange(start_index, 9999999):
            if x == len_data:
                return None, None


            sma_stop_loss = self.sma[x] * (1 - self.k['stop_loss_abs_pct_loss'])

            date_today = self.stock_2_trimmed[x]['Date']
            current_price = self.closes[x]
            price_log.append(current_price)
            date_log.append(date_today)
            price_change_pc = (current_price - result.entry_price) / result.entry_price

            if trading_up:
                ret = price_change_pc
            else:
                ret = -price_change_pc


            ### print x, result.stock_2, sigma_span[x], result.entry_price, current_price


            current_date = self.stock_2_trimmed[x]['Date']
            current_date = datetime.datetime.strptime(current_date, '%Y-%m-%d')
            current_year = current_date.year
            current_year_entry_date = datetime.datetime(current_year, self.k['entry_month'], self.k['entry_day'])
            current_year_exit_date = datetime.datetime(current_year, self.k['exit_month'], self.k['exit_day'])

            score = 0
            
            if current_date > current_year_entry_date or current_date < current_year_exit_date:
                score += 0

            # if self.closes[x] < self.sma[x] and self.closes[x-1] < self.sma[x-1]:
            if self.closes[x] < self.sma[x]:
                try:
                    # if current_date.weekday() > datetime.datetime.strptime(self.stock_2_trimmed[x+1]['Date'], '%Y-%m-%d').weekday():
                    if current_date.month != datetime.datetime.strptime(self.stock_2_trimmed[x+1]['Date'], '%Y-%m-%d').month:
                        score += 1
                    elif self.closes[x] < sma_stop_loss:
                        score += 1
                except:
                    pass

            # print start_index, score, current_date, current_year_entry_date, current_year_exit_date
                
#            if trading_up and self.closes[x] <= self.sma[x]:
            if trading_up and \
               (score > 0 or ret < pc_stop_loss):

                result.time_in_trade = x - (start_index - 1)
                result.exit_price = current_price
                result.ret = ret
                result.chained_ret = 1 + ret
                result.exit_date = date_today
                result.exit_rsi = None
                result.end_index = x
                result.price_log = price_log
                result.date_log = date_log

                if ret > 0:
                    # print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Profit"
                else:
                    # print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Loss"

                return result, result.end_index


            elif trading_down and (time_in == exit_time or \
                current_price < result.target or \
                current_price > (result.entry_price + (stop_loss * entry_sigma)) or \
                ret <= pc_stop_loss or \
                (ret < 0 and time_in > exit_after_loss)):

                result.time_in_trade = x - (start_index - 1)
                result.exit_price = current_price
                result.ret = ret
                result.chained_ret = 1 + ret
                result.exit_date = date_today
                result.exit_rsi = None
                result.end_index = x
                result.price_log = price_log
                result.date_log = date_log

                if ret > 0:
                    # print "Profit: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Profit"

                else:
                    #print "Loss: ", result.entry_date, date_today, result.entry_price, current_price, ret, '\n'
                    result.trade_result = "Loss"

                return result, result.end_index

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

        result.entry_score = result.entry_sigma_over_p

        return result









class trade_result(object):

    def __init__(self):
        self.stock_1 = None
        self.stock_2 = None

        self.entry_date = None
        self.exit_date = None
        self.start_index = None
        self.end_index = None
        self.long_short = ''

        self.entry_price = None
        self.exit_price = None
        self.entry_vol = None
        self.entry_rsi = None
        self.exit_rsi = None
        self.entry_sma = None
        self.entry_sigma = None
        self.entry_sigma_over_p = None

        self.entry_sigma_percentile = None

        self.price_log = []

        self.ret = None
        self.chained_ret = None
        self.time_in_trade = None
        self.trade_result = '' # profit, loss, timeout

