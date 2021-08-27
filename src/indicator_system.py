import numpy as np
import scipy as scipy
import datetime
import time
import logging

import math
# import statsmodels.tsa.stattools as stats

import toolsx as tools

from util.memoize import memoize, MemoizeMutable

from data.redis import manage_redis
# from src import math_tools
from src.cointegrations_data import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly, \
                                    get_bunches_of_pairs

import src.signals.signals as signals


def run_indicator_system():
    sectors = ('basic_materials', 'conglomerates', 'consumer_goods', 'financial', 'healthcare', 'industrial_services', \
               'services', 'technology', 'utilities')

    in_file_name = 'etfs_etns_sp_500'
    # in_file_name = 'big_etfs'
    location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + in_file_name + '.csv'
    in_file = open(location, 'r')
    stock_list = in_file.read().split('\n')
    for k, item in enumerate(stock_list):
        new_val = item.split('\r')[0]
        stock_list[k] = new_val
    in_file.close()

    etf_list = get_etf_list()

    # stock_list = ['SPY', 'TLT']
    paired_list = get_paired_stock_list(sorted(stock_list), fixed_stock='SPY')

    len_stocks = len(paired_list)

    interest_items = []
    trade_log = []

    # current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_file_name = '/home/wilmott/Desktop/fourseasons/fourseasons/results/indicator_results_' + in_file_name + '_' + str(current_time) +'.csv'
    out_file = open(out_file_name, 'w')

    data_failures = []

    days_analyzed = 0
    base_stock = paired_list[0]['stock_1']
    stock_1_data = manage_redis.parse_fast_data(base_stock, db_to_use=1)
    for k, item in enumerate(paired_list):
        is_stock = not item['stock_2'] in etf_list
        print k, len_stocks, item['stock_1'], item['stock_2'], '\t', is_stock
        output, trades, x = do_indicator_test(item, k, len(paired_list), stock_1_data, is_stock = is_stock)
        if x:
            days_analyzed += x
        if trades is not None and len(trades) > 0:
            trade_log.extend(trades)
        if output == 'data_failure' and trades == None and x == None:
            data_failures.append(item['stock_2'])

    print "Data Failures: ", len(data_failures), data_failures

    output_fields = ['stock_1', 'stock_2', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'long_short', 'entry_vol', 'entry_sma', \
                     'entry_rsi', 'exit_rsi', 'entry_sigma', 'entry_sigma_over_p', 'time_in_trade', \
                     'trade_result', 'ret', 'chained_ret']


    do_backtest = 'y'

    # The backtest_trade_log effectively shrinks the trade log into only those trades that would be
    # possible in a chronologically traded system (i.e. one at a time)
    total_trades_available = len(trade_log)
    ###
    if do_backtest == 'y':
        trade_log = backtest_trade_log(trade_log)
    elif do_backtest == 'm':
        trade_log = multi_port_backtest_trade_log(trade_log, num_logs = 4)
        backtest_only_fields = ['entry_dollars', 'exit_dollars', 'start_cash_avail', 'total_valuation']

        output_fields.extend(backtest_only_fields)
    ###


    output_string = ','.join(output_fields)
    output_string += '\n'

    rets = []
    for trade_item in trade_log:
        rets.append(trade_item.chained_ret)

#		print trade_item.stock_2, trade_item.entry_date, trade_item.exit_date, trade_item.entry_price, trade_item.exit_price, trade_item.ret, trade_item.chained_ret, \
#				trade_item.entry_sigma, trade_item.entry_sigma_over_p
#		if trade_item.chained_ret < 0.0:
#			return

        output_string += ','.join([str(getattr(trade_item, item)) for item in output_fields])
        output_string += '\n'

    out_file.write(output_string)
    out_file.close()

    # Note that if there is a negative total_return, then the pow function will throw a domain error!!!!
    total_return = np.product(rets)
    geom_return = math.pow(total_return, (1.0/len(trade_log)))

    ###
    sharpe_ratio, sortino_ratio, total_days_in = get_intra_prices(trade_log)
    total_years_in = total_days_in / 252.0
    annualized_return = math.pow(total_return, (1.0/total_years_in))


    negative_ret_list = [r for r in rets if r <= 1.0]
    positive_ret_list = [r for r in rets if r > 1.0]

    total_losers = len(negative_ret_list)
    total_winners = len(positive_ret_list)
    pct_losers = float(total_losers) / len(rets)

    avg_loser = np.mean(negative_ret_list)
    avg_winner = np.mean(positive_ret_list)
    profit_factor = ((1-pct_losers) * abs(avg_winner-1)) / ((pct_losers) * abs(avg_loser-1))

    print "\nTrade PctLosers, Mu Winner, Mu Loser, PF: ", round(pct_losers, 5), round(avg_winner, 5), round(avg_loser, 5), round(profit_factor, 5)
    print "\nTrades, Total, geom ret, ann ret", len(trade_log), round(total_return, 5), round(geom_return, 5), round(annualized_return, 5)
    ###

    print '\nStock-Days Analyzed', days_analyzed
    print 'Total Trades Available: ', total_trades_available

    print "\nFinished: ", len_stocks
    print "File Written: ", in_file_name + out_file_name.split(in_file_name)[-1]

def do_indicator_test(item, k, len_stocks, stock_1_data, is_stock):
#	stock_1_data = manage_redis.parse_fast_data(item['stock_1'], db_to_use=0)
    stock_2_data = manage_redis.parse_fast_data(item['stock_2'], db_to_use=1)

    try:
        # print "Getting data for: ", item['stock_1'], item['stock_2']
        stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
    except:
        logging.warning('Exception in getting corrected data: %s' % (item['stock_2']))
        return 'data_failure', None, None

    stock_1_volume = [x['Volume'] for x in stock_1_trimmed]

    stock_2_close = [x['AdjClose'] for x in stock_2_trimmed]
    # stock_2_high = [x['AdjHigh'] for x in stock_2_trimmed]
    # stock_2_low = [x['AdjLow'] for x in stock_2_trimmed]
    stock_2_volume = [x['Volume'] for x in stock_2_trimmed]

    if len(stock_2_trimmed) < 253:
        return None, None, None

    days_analyzed = len(stock_2_trimmed) - 252

    trade_log = []

    end_data = len(stock_2_trimmed)
    output = None
    next_index = 0

    # signal = signals.MovingAverageSeasonalitySystem(stock_2_close, stock_2_volume, stock_2_trimmed, item, is_stock=is_stock, kwargs = {'stock_1_close':stock_1_close} )
    # signal = signals.SignalsSigmaSpanVolatilityTest_2(stock_2_close, stock_2_volume, stock_2_trimmed, item, is_stock=is_stock)
    # signal = signals.SignalsSigmaSpanVolatilityTest_3(stock_2_close, stock_2_volume, stock_2_trimmed, item, is_stock=is_stock, kwargs = {'stock_1_close':stock_1_close} )
    # signal = signals.InWeekAfterDown(stock_1_close, stock_1_volume, stock_1_trimmed, item)
    signal = signals.ATRExitSignal(stock_1_close, stock_1_volume, stock_1_trimmed, item)
    
    for x in xrange(252, end_data):
        # If we've been told we're still in a trade then we simply skip this day
        if x <= next_index:
            continue

        trade_result = False
        result = None
        trade_result = signal.get_entry_signal(x)
        if not trade_result:
            continue
        elif trade_result:
            # We will only enter here if get_entry_signal() returned a trade_result, meaning that it signaled an entry
            # and filled in the entry parameters for us
            result = trade_result
            result, next_index = signal.get_exit(x, result)

            if result:
                trade_log.append(result)


    return output, trade_log, days_analyzed


def backtest_trade_log(trade_log, exclude_log=None):
    print "Total Entries Found Length: ", len(trade_log)
    start_date = datetime.datetime(1900, 1, 1)
    end_date = datetime.datetime(2100,1,1)
    # chrono_trade_log = [t for t in trade_log if datetime.datetime.strptime(t.entry_date, '%Y-%m-%d') > start_date and datetime.datetime.strptime(t.entry_date, '%Y-%m-%d') < end_date]
    chrono_trade_log = [t for t in trade_log if datetime.datetime.strptime(t.entry_date, '%Y%m%d') > start_date and datetime.datetime.strptime(t.entry_date, '%Y%m%d') < end_date]
    chrono_trade_log.sort(key=lambda x: x.entry_date)

    # for result in chrono_trade_log:
        # print result.stock_2, result.entry_date, result.exit_date

    small_log = []
    len_data = len(chrono_trade_log)

    x = 0
    while x < len_data:

        current_date = chrono_trade_log[x].entry_date

        # We must skip trading opportunities if we have not exited the first trade!!
        if len(small_log) > 0:
            last_exit = small_log[-1].exit_date
            todays_entry = current_date

            # last_exit = datetime.datetime.strptime(last_exit, '%Y-%m-%d')
            # todays_entry = datetime.datetime.strptime(todays_entry, '%Y-%m-%d')

            last_exit = datetime.datetime.strptime(last_exit, '%Y%m%d')
            todays_entry = datetime.datetime.strptime(todays_entry, '%Y%m%d')

            # The continue here will cut out all other logic and move on to the next item in the
            # master trade log. Basically, if we are already in a trade and have not exited yet,
            # then nothing else matters, we must keep looking at other trades
            if todays_entry <= last_exit:
                x += 1
                continue

        # We are iterating over the entry date-sorted chrono trade log so we are guaranteed to
        # only encounter trades in chrono order. The filter here finds ALL trades starting on the
        # day of the current trade, so that we can see all possibilities like in real life
        # Then append the best trade opportunity and incrememt the chrono trade log counter by the
        # number of trades that started today

        start_today = [z for z in chrono_trade_log if z.entry_date == current_date]
        # print "start_today", current_date, len(start_today)

        # Choose the most volatile stock at a given day
        target = 999

        ### start_today.sort(key=lambda z: abs((z.entry_score - target)), reverse=False)
        # start_today.sort(key=lambda z: abs((z.entry_sigma_over_p - target)), reverse=False)
        # start_today.sort(key=lambda z: abs((z.entry_volatility - target)), reverse=False)

        ### Executing this results in selecting a *random* entry that is available on the given day
#        import random
#        idx = random.randrange(len(start_today))
#        start_today = [start_today[idx]]

        # print "here", [z.entry_sigma_over_p for z in start_today]

        if len(start_today) > 0:
            if exclude_log is not None:
                # This will avoid entering more than x number of trades on the same day. This reduces the timing risk
                exclude_log_start_today = [e for e in exclude_log if e.entry_date == current_date]
                for s in xrange(0, len(start_today)):
                    if start_today[s] not in exclude_log and len(exclude_log_start_today) <= 0:
                        small_log.append(start_today[s])
                        break
            else:
                small_log.append(start_today[0])
        else:
            pass
            # small_log.append(start_today[0])
        ###
        print "Start Today: ", len(small_log), len(start_today)

        x += len(start_today)

    return small_log

def get_intra_prices(trade_log):

    # If backtest trade log was called first, this is already sorted, but if not, we must do it here
    trade_log.sort(key=lambda x: x.entry_date)

    ### To properly determine the sharpe ratio, we must compare the returns in the trade log with returns in
    # SPY over the same period of time. Specifically, we must know the number of days that SPY traded within the
    # time period of interest, then insert returns of zero (1.0) in the trade log so that the length matches SPY
    ref_price_data = manage_redis.parse_fast_data('SPY', db_to_use=1)

    system_ret_log = []
    print "*****************************SHARPE RATIO ANALYSIS"
    for item in trade_log:
        # print item.entry_date, item.exit_date, item.entry_price, item.exit_price, item.ret, item.time_in_trade, len(item.price_log), item.price_log
        p = ('new_trade', item.long_short, item.price_log[0])
        system_ret_log.append(p)
        if len(item.price_log) > 1:
            for x in xrange(1, len(item.price_log)):
                p = ('existing_trade', item.long_short, item.price_log[x])
                system_ret_log.append(p)

    # print '\n\n\n', system_ret_log[-20:0], len(system_ret_log)


    first_entry = trade_log[0].entry_date
    last_exit = trade_log[-1].exit_date

    # first_entry = datetime.datetime.strptime(first_entry, '%Y-%m-%d')
    # last_exit = datetime.datetime.strptime(last_exit, '%Y-%m-%d')
    first_entry = datetime.datetime.strptime(first_entry, '%Y%m%d')
    last_exit = datetime.datetime.strptime(last_exit, '%Y%m%d')

    ref_trimmed_price_data = []
    for day in ref_price_data:
        # z = datetime.datetime.strptime(day['Date'], '%Y-%m-%d')
        z = datetime.datetime.strptime(day['Date'], '%Y%m%d')
        if z >= first_entry and z <= last_exit:
            # We hardcode "long" here because this is a buy and hold assumption...
            p = ('existing_trade', 'long', day['AdjClose'])
            ref_trimmed_price_data.append(p)

    rmean, rstd, rneg_std, rpos_std, rsharpe, rsortino, ravg_loser, ravg_winner, rpct_losers = get_sharpe_ratio(ref_trimmed_price_data)
#    print "Reference: \nArith Mu, Sigma, Neg Sigma, Sharpe, Sortino, #Days:", round(mean, 6), round(std, 6), round(neg_std, 6), round(sharpe, 6), round(sortino, 6), len(ref_trimmed_price_data)

    smean, sstd, sneg_std, spos_std, ssharpe, ssortino, savg_loser, savg_winner, spct_losers = get_sharpe_ratio(system_ret_log)
#    print "\nSystem: \nArith Mu, Sigma, Neg Sigma, Sharpe, Sortino, #Days:", round(mean, 6), round(std, 6), round(neg_std, 6), round(sharpe, 6), round(sortino, 6), len(system_ret_log)


    print '\t\tSystem:\t\tReference:'
    print 'ArithMu: \t', round(smean, 6), '\t', round(rmean, 6)
    print 'Sigma: \t\t', round(sstd, 6), '\t', round(rstd, 6)
    print 'NegSigma: \t', round(sneg_std, 6), '\t', round(rneg_std, 6)
#    print 'PosSigma: \t', round(spos_std, 6), '\t', round(rpos_std, 6)
    print 'NegSigma/Tot: \t', round((sneg_std/sstd), 6), '\t', round((rneg_std/rstd), 6)
    print 'Sharpe: \t', round(ssharpe, 6), '\t', round(rsharpe, 6)
    print 'Sortino: \t', round(ssortino, 6), '\t', round(rsortino, 6)

    # profit_factor = ((1-spct_losers) * (savg_winner)) / ((spct_losers) * savg_loser)
    # print "\nDay By Day PctLosers, Mu Winner, Mu Loser, PF: ", round(spct_losers, 5), round(savg_winner, 5), round(savg_loser, 5), round(profit_factor, 5)
    print "Frac Time In Market: ", round(float(len(system_ret_log)) / len(ref_trimmed_price_data), 4)

    total_days = len(ref_trimmed_price_data)

    return ssharpe, ssortino, total_days


def get_sharpe_ratio(price_list):

    # The input argument into this function is a list of 3-tuples of: ( new/existing trade , long/short, price )

    ret_list = []

    len_data = len(price_list)

    for k in xrange(0, len_data):
        if k == 0:
            continue

        if price_list[k][0] == 'new_trade':
            # we skip adding a "return" for the first day of a trade since we have not been in the trade overnight yet

            current_entry_price = price_list[k][2]
#			print k, price_list[k], (price_list[k][2] / price_list[k-1][2])
            continue

        # Strategy: Calculate baseline ret same safe way as always. If short, change sign as we did before. Then add
        # one to use chained return
        baseline_ret = (price_list[k][2] - price_list[k-1][2]) / price_list[k-1][2]
        if price_list[k][1] == 'short':
            baseline_ret = -baseline_ret
        current_ret = baseline_ret + 1

        ret_list.append(current_ret)
#		print k, price_list[k], current_ret


    # This changed recently: Here, we use the same length of the list, but put in a zero if the value is not the same
    # sign as we care about
    negative_ret_list = [min(r, 1.0) for r in ret_list]
    positive_ret_list = [max(r, 1.0) for r in ret_list]

    total_losers = len(negative_ret_list)
    total_winners = len(positive_ret_list)
    pct_losers = float(total_losers) / len(ret_list)

    avg_loser = np.mean(negative_ret_list)
    avg_winner = np.mean(positive_ret_list)

    mean = np.mean(ret_list) - 1
    std = np.std(ret_list)
    neg_std = np.std(negative_ret_list)
    pos_std = np.std(positive_ret_list)
    sharpe_ratio = mean / std
    sortino_ratio = mean / neg_std

    return mean, std, neg_std, pos_std, sharpe_ratio, sortino_ratio, avg_loser, avg_winner, pct_losers

def multi_port_backtest_trade_log(trade_log, num_logs = 1):

    final_log = []

    # num_logs = 1

    for n in xrange(0, num_logs):
        this_log = backtest_trade_log(trade_log, exclude_log=final_log)
        final_log.extend(this_log)

    final_log.sort(key=lambda x: x.exit_date)
    final_log.sort(key=lambda x: x.entry_date)

    ###
    print "lengths: ", len(trade_log), len(final_log)

    most_recent_port_value = 1.0
    cash_available = 1.0
    trades_open_now = 0
    trade_slots_open = num_logs
    pct_cash_per_trade = 1 / num_logs

    first_entry_date = datetime.datetime.strptime(final_log[0].entry_date, '%Y-%m-%d')
    last_exit_date = datetime.datetime.strptime(final_log[len(final_log) - 1].exit_date, '%Y-%m-%d')

    ###
    print "Dates: ", first_entry_date, last_exit_date

    current_date = first_entry_date

    fmt = '%Y-%m-%d'

    open_trades = []
    output_log = []

    while current_date <= last_exit_date:

        todays_date = current_date.strftime(fmt)

        print "\nCurrent Date, master loop: ", todays_date

        starts_today = [z for z in final_log if z.entry_date == todays_date]
        exits_today = [z for z in final_log if z.exit_date == todays_date]

        if len(starts_today) == 0 and len(exits_today) == 0:
            current_date += datetime.timedelta(days=1)
            continue

        if starts_today:
            print "Starts Today: ", starts_today
        if exits_today:
            print "Exits Today: ", exits_today

        print "Trade Slots Open: ", trade_slots_open


        trade_slots_open = num_logs - len(open_trades)
        if len(exits_today) > 0:

            for x in xrange(0, len(exits_today)):

                current_trade = exits_today[x]

                try:
                    print "removing this: ", current_trade
                    print "open trades before: ", open_trades
                    open_trades = remove_trade(open_trades, current_trade)
                    print "open trades after: ", open_trades

                except:
                    print "could not find this trade"


                current_trade.exit_dollars = current_trade.entry_dollars * current_trade.chained_ret
                cash_available += current_trade.exit_dollars
                print "Current Trade Exit Dollars: ", current_trade.exit_dollars

                print "cash available: ", cash_available

                current_trade.total_valuation = get_valuation(open_trades, cash_available, todays_date)

                output_log.append(current_trade)


        trade_slots_open = num_logs - len(open_trades)
        if len(starts_today) > 0 and trade_slots_open > 0:

            pct_cash_per_trade = 1.0 / trade_slots_open
            amount_to_allocate = cash_available * pct_cash_per_trade

            max_loop = min(trade_slots_open, len(starts_today))
            for x in xrange(0, max_loop):
                starts_today[x].entry_dollars = amount_to_allocate
                starts_today[x].exit_dollars = 'not_set'
                starts_today[x].start_cash_avail = amount_to_allocate / pct_cash_per_trade
                starts_today[x].profit_loss = 'not_set'

                cash_available -= amount_to_allocate

                open_trades.append(starts_today[x])



        # start_today.sort(key=lambda z: abs((z.entry_score - target)), reverse=False)

        # print current_date, todays_date, starts_today

        current_date += datetime.timedelta(days=1)

#    current_chained_total = 1.0
#
#    for x in xrange(0, len(final_log)):
##        item.ret = (item.ret / num_logs)
##        item.chained_ret = ((item.chained_ret - 1) / num_logs) + 1
#
#        final_log[x].ret = (final_log[x].ret / num_logs)
#        final_log[x].chained_ret = ((final_log[x].chained_ret - 1) / num_logs) + 1
#
#        if x < 2:
#            final_log[x].product_ret = final_log[x].chained_ret * current_chained_total
#
#        else:
#            final_log[x].product_ret = final_log[x].chained_ret * final_log[x-2].chained_ret
#
#    if num_logs > 1:
#        # Here we must hack some statistical values to print so we can easily chart the result...
#        for x in xrange(0, len(final_log)):
#            current_chained = 1
#            current_average = 1
            


    return output_log




def remove_trade(open_trades, current_trade):

    for x in xrange(0, len(open_trades)):
        item = open_trades[x]

        if item.stock_2 != current_trade.stock_2:
            continue
        if item.entry_date != current_trade.entry_date:
            continue
        if item.entry_price != current_trade.entry_price:
            continue

        open_trades.pop(x)

    return open_trades


def get_valuation(open_trades, cash_available, current_date):
    # take the latest item in the output log as the most recent portfolio total valuation
    # cash available should already reflect the cash received from the current_trade
    # that is, we don't actually need the current_trade here
    # then, add in the valuation at this point in time for the open_trades that exist
    # as a naive first-pass, we could value the open trades as their entry dollar value, though this would not be
    # quite accurate

#    fmt = '%Y-%m-%d'
#    todays_date = current_date.strftime(fmt)

    todays_date = current_date

    total_valuation = cash_available

    for item in open_trades:

        entry_price = item.entry_price

        i = item.date_log.index(todays_date)
        current_price = item.price_log[i]

        current_return = current_price / entry_price

        current_trade_value = item.entry_dollars * current_return

        total_valuation += current_trade_value


    return total_valuation







def get_etf_list():
    in_file_name = 'etfs_etns'
    location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + in_file_name + '.csv'

    in_file = open(location, 'r')
    etf_list = in_file.read().split('\n')
    for k, item in enumerate(etf_list):
        new_val = item.split('\r')[0]
        etf_list[k] = new_val
    in_file.close()

    return etf_list

class StatsItems(object):

    def __init__(self):

        total_trades = None
        num_winners = None
        num_losers = None

        arith_mean_all = None
        geom_mean_all = None
        chained_all = None
        arith_mean_winners = None
        arith_mean_losers = None

        sigma_ret_all = None
        sigma_ret_winners = None
        sigma_ret_losers = None

        sharpe_ratio = None
        sortino_ratio = None




