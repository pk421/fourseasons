import numpy as np
import scipy as scipy
import datetime

import math
import itertools

import src.toolsx as tools

from data.redis                                     import manage_redis
from src.data_retriever                             import load_redis
from src.data_retriever                             import multithread_yahoo_download

from src.indicator_system                           import get_sharpe_ratio
from src.portfolio_analysis.portfolio_utils         import get_data, get_sweep_data
from src.portfolio_analysis.portfolio_constants     import custom_assets_list, live_portfolio

from src.math_tools                                 import get_returns, get_ln_returns

import multiprocessing
import Queue
import sys

import logging

# determine whether to download new data from the internet
UPDATE_DATA=False
logging.root.setLevel(logging.CRITICAL)

def do_optimization(mdp_port, x):
    theoretical_weights = mdp_port.get_tangency_weights(x)

    # theoretical_weights = np.array([ [1.0 / len(mdp_port.assets)] for x in mdp_port.assets])
    # theoretical_weights = np.array([ [0.30], [0.15], [0.40], [0.075], [0.075] ])
    # theoretical_weights = np.array([ [0.30], [0.55], [0.15] ])

    if logging.root.level < 25:
        print "Theoretical Result: \n", theoretical_weights
        print "weighted vols equal one check: ", mdp_port.weighted_vols_equal_one(theoretical_weights)

    # This section should not be necessary since weights are already normalized. It's an added safety.
    abs_weight_sum = np.sum([abs(n) for n in theoretical_weights])
    real_weight_sum = np.sum(theoretical_weights)
    weight_ratio = real_weight_sum / abs_weight_sum if round(abs_weight_sum, 6) != 1.0 else 1.0

    normalized_theoretical_weights = np.array([ n * weight_ratio for n in theoretical_weights])

    # Turn this on to fix the weights of at each rebalance
    ### HACK:
    # normalized_theoretical_weights = np.array([[0.3], [0.15], [0.4], [0.075], [0.075]])

    # MDP Optimized For 2338/2339 historical days single rebalance
    # normalized_theoretical_weights = np.array([[0.25], [0.28], [0.29], [0.08], [0.10]])

    # Naive Risk Parity Optimized for 2338/2339 historical days single rebalance
    # normalized_theoretical_weights = np.array([[0.14], [0.40], [0.19], [0.13], [0.14]])

    return mdp_port, normalized_theoretical_weights

def worker_do_analysis_no_sharing(input_combos, dates_dict, y, lock, redis_historical_data_dict, stream_out):
    """
    Does not share data between threads. The resulting data is dumped in a csv file and concatenated at the end. Much
    more efficient than attempting to share data between threads.
    """
    results_dict = {'output_string':'', 'all_stats_list':[]}
    for x in xrange(0, len(input_combos)):
        assets = input_combos[x]

        # start_date = 20050101
        end_date = 20150612
        start_date = end_date - 64

        skip_iter = False
        for item in assets:
            if int(dates_dict[item]['start_date']) > start_date or int(dates_dict[item]['end_date']) < end_date:
                skip_iter = True
                break
        if skip_iter:
            with lock:
                stream_out.write('\nItems Left: ' + str(len(input_combos) - x) + ' Dates skipping: ' + str(assets))
            continue

        system_stats = do_analysis(assets, write_to_file=False, historical_data_dict=redis_historical_data_dict)
        with lock:
            results_dict['all_stats_list'].append(system_stats)
            # "Items: ", k, ' of ', len(combinations), \
            screen_output =   "\nItems Left: " + str(len(input_combos) - x) + \
                              " AnnRet: " + str(round(system_stats['annualized_return'], 6)) + \
                              " Sigma: " + str(round(system_stats['sigma'], 6)) + \
                              " Mu DR: " + str(round(system_stats['mean_diversification_ratio'], 6))
                              # "Sharpe: ", round(system_stats['sharpe'], 6), \
            stream_out.write(screen_output)

            results_dict['output_string'] += str(round(system_stats['annualized_return'], 6)) + ','
            results_dict['output_string'] += str(round(system_stats['sharpe'], 6)) + ','
            results_dict['output_string'] += str(round(system_stats['sigma'], 6)) + ','
            results_dict['output_string'] += str(round(system_stats['mean_diversification_ratio'], 6)) + ','
            results_dict['output_string'] += ','.join([str(a) for a in assets])
            results_dict['output_string'] += '\n'

    out_file_name = '/home/wilmott/Desktop/fourseasons/fourseasons/results/worker_temp' + '_' + str(y) +'.csv'
    with open(out_file_name, 'w') as f:
        f.write(results_dict['output_string'])

def run_sweep_portfolios():

    # stock_list = open('/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + 'etfs_for_sweep.csv', 'r')
    # assets = stock_list.read().rstrip().split('\n')

    assets = custom_assets_list

    # This section basically essentially caches the data rather than calling a get_data and re-parsing each time in
    # portfolio_utils
    # add these two assets just for the purpose of retrieving data. They are used for the "reference" system
    additional_assets = ['IWM', 'TLT']
    modified_assets = list(set(assets + additional_assets))

    # multithread_yahoo_download(thread_count=20, update_check=False, \
    #                            new_only=False, store_location = 'data/portfolio_analysis/', use_list=modified_assets)
    # load_redis(stock_list='tda_free_etfs.csv', db_number=1, file_location='data/portfolio_analysis/', dict_size=3, use_list=modified_assets)
    redis_historical_data_dict = {}
    dates_dict = {}
    for item in modified_assets:
        redis_historical_data_dict[item] = manage_redis.parse_fast_data(item, db_to_use=1)

        start_date = redis_historical_data_dict[item][0]['Date']
        end_date = redis_historical_data_dict[item][-1]['Date']
        dates_dict[item] = {'start_date': start_date, 'end_date': end_date}

        print "Asset, StartDate, EndDate: ", item, '\t', start_date, '\t', end_date

    combinations = []
    combos = [ itertools.combinations(assets, 6), itertools.combinations(assets, 5), itertools.combinations(assets, 4)]
    # combos = [ itertools.combinations(assets, 3) ]

    for combo in combos:
        try:
            while True:
                c = combo.next()
                cleaned_c = [b.strip() for b in c]
                combinations.append(cleaned_c)
        except:
            pass

    print "Total Combinations: ", len(combinations)

    output_string = 'Ann Return, Sharpe Ratio, Sigma, DR, Assets\n'

    lock = multiprocessing.Lock()

    num_threads = 5
    process_list = []

    work = []
    work_breaks = int(len(combinations) / num_threads)

    for x in xrange(0, (num_threads-1)):
        work.append(combinations[x*work_breaks:(x+1)*work_breaks])
    work.append(combinations[(num_threads-1)*work_breaks:])
    for x in xrange(0, num_threads):
        process_list.append(multiprocessing.Process(target=worker_do_analysis_no_sharing, args=(work[x], dates_dict, x, lock, redis_historical_data_dict, sys.stdout)))
    for x in xrange(0, num_threads):
        process_list[x].start()
    for x in xrange(0, num_threads):
        process_list[x].join()

    for x in xrange(num_threads):
        worker_file = '/home/wilmott/Desktop/fourseasons/fourseasons/results/worker_temp' + '_' + str(x) +'.csv'
        with open(worker_file, 'r') as f:
            output_string += f.read()


    current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_file_name = '/home/wilmott/Desktop/fourseasons/fourseasons/results/sweep_portfolio_analysis' + '_' + str(current_time) +'.csv'
    with open(out_file_name, 'w') as f:
        f.write(output_string)
    print "\n\nFile Written: ", out_file_name.split('/')[-1]

    return

def do_analysis(assets=None, write_to_file=True, historical_data_dict={}):

    assets_list = assets

    mdp_port = MDPPortfolio(assets_list=assets_list)
    logging.debug(str(mdp_port.assets))
    mdp_port = get_sweep_data(mdp_port, base_etf=mdp_port.assets[0], last_x_days=64, get_new_data=UPDATE_DATA, historical_data=historical_data_dict)

    mdp_port.normalized_weights = np.array([ [1.0 / len(mdp_port.assets)] for x in mdp_port.assets])
    mdp_port.current_weights = mdp_port.normalized_weights

    # Start with 1000 dollars and buy appropriate number of shares in each item
    starting_valuation = 1000
    mdp_port.shares = [ (starting_valuation * mdp_port.normalized_weights[k][0] / mdp_port.closes[v][0]) for k, v in enumerate(mdp_port.assets) ]
    mdp_port.current_entry_prices = [ mdp_port.closes[v][0] for k, v in enumerate(mdp_port.assets) ]

    x=0
    mdp_port.x = x
    mdp_port.lookback = 63
    mdp_port.rebalance_time = 63
    mdp_port.rebalance_counter = 0
    mdp_port.rebalance_now = False
    mdp_port.rebalance_log = []
    while x < len(mdp_port.trimmed[mdp_port.assets[0]]):
        if not mdp_port.rebalance_now:

            mdp_port.x = x
            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)
            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation])
            if logging.root.level < 25:
                print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation
            mdp_port.rebalance_counter += 1

            if x >= mdp_port.rebalance_time:
                # The statement below DOES have an impact on whether a rebalance is hit in this if block
                # It also affects overall program flow and the end result
                _ = mdp_port.get_tangency_weights(x)
                trailing_diversification_ratio = mdp_port.get_diversification_ratio(weights='current')
                mdp_port.trailing_DRs.append(trailing_diversification_ratio)
                rebalance_date = mdp_port.trimmed[mdp_port.assets[0]][x]['Date']
                if logging.root.level < 25:
                    print "Trailing DR: ", x, trailing_diversification_ratio

                # Only consider a rebalance after rebalance_time, if div ratio is low, rebalance. If it's high, wait
                # another rebalance_time to check again
                if mdp_port.rebalance_counter >= mdp_port.rebalance_time and trailing_diversification_ratio < 10.0:
                    mdp_port.rebalance_now = True
                elif mdp_port.rebalance_counter >= mdp_port.rebalance_time and trailing_diversification_ratio >= 10.0:
                    mdp_port.rebalance_now = False
                    # mdp_port.rebalance_counter = 0
                elif trailing_diversification_ratio <= 1.0:
                    mdp_port.rebalance_now = True
            else:
                mdp_port.trailing_DRs.append(0)
            x += 1
            continue
        else:
            mdp_port.rebalance_counter = 0
            mdp_port.rebalance_now = False
            # When calculating the rebalance ratio, we first assume we used the old weights for today's valuation. Then
            # we calculate new weights for today, value the portfolio for today, then find the ratio for today if we
            # had used the old weights for today
            old_weighted_valuation = get_port_valuation(mdp_port, x=x)
            mdp_port.x = x

            rebalance_old_div_ratio = mdp_port.get_diversification_ratio(weights='current')

            mdp_port, theoretical_weights = do_optimization(mdp_port, x)
            
            ###
            if logging.root.level < 25:
                print "\n\n\nTheoretical Weights: ", str(theoretical_weights)
            mdp_port.set_normalized_weights(theoretical_weights, old_weighted_valuation, x)
            ###

            # calling get_port_valuation here is used to set the current weights, used in the div ratio below
            _ = get_port_valuation(mdp_port, x=x)
            for k, asset in enumerate(mdp_port.assets):
                if round(mdp_port.normalized_weights[k], 6) != round(mdp_port.current_weights[k], 6):
                    print "\n", mdp_port.assets
                    print "\n", mdp_port.normalized_weights
                    print "\n", mdp_port.current_weights
                    raise Exception("Normalized weights do not equal the current weights after setting normalized.")

            trailing_diversification_ratio = mdp_port.get_diversification_ratio(weights='normalized')
            mdp_port.trailing_DRs.append(trailing_diversification_ratio)
            if logging.root.level < 25:
                print "New Trailing Diversification Ratio: ", x, trailing_diversification_ratio
            mdp_port.rebalance_log.append((rebalance_date, rebalance_old_div_ratio, trailing_diversification_ratio, theoretical_weights))

            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)

            if round(current_portfolio_valuation, 6) != round(old_weighted_valuation, 6):
                raise Exception("Current Valuation != Old Weighted Valuation: Curr/Old: " + str(current_portfolio_valuation) + '\t' + str(old_weighted_valuation))

            if logging.root.level < 25:
                print "Current Portfolio New Valuation: ", current_portfolio_valuation

            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation ])
            if logging.root.level < 25:
                print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation
            x+=1

    for i, p in enumerate(mdp_port.portfolio_valuations):
        if i == 0:
            starting_value = p[1]
        mdp_port.portfolio_valuations[i][1] = mdp_port.portfolio_valuations[i][1] / starting_value

    if len(mdp_port.trimmed[mdp_port.assets[0]]) != len(mdp_port.portfolio_valuations):
        raise Exception('Length of asset list data does not match length of valuations data.')

    system_results = aggregate_statistics(mdp_port, write_to_file=write_to_file, historical_data=historical_data_dict)
    return system_results


def aggregate_statistics(mdp_port, write_to_file=True, historical_data={}):

    if logging.root.level < 25:
        print '\n'.join([str(r) for r in mdp_port.rebalance_log])

    ### Create sharpe_price_list and a Moving Average of port valuations. Last if statement will simply store days
    # the valuation was < MA
    sharpe_price_list = []
    sys_closes = []
    system_dma = []
    trade_days_skipped = []
    for k, val in enumerate(mdp_port.portfolio_valuations):
        sharpe_price_list.append(('existing_trade', 'long', val[1]))
        sys_closes.append(val[1])
        if k < 200:
            system_dma.append(1)
        else:
            system_dma.append(np.mean([n[1] for n in mdp_port.portfolio_valuations[k-200:k+1]]))
        if mdp_port.portfolio_valuations[k-1][1] < system_dma[k-1]:
            trade_days_skipped.append(k)

    ### Show starting and ending valuation (after potentially adding a moving average filter)
    if logging.root.level < 25:
        print '\n', mdp_port.portfolio_valuations[0], mdp_port.portfolio_valuations[-1]

    ### Get Ref Data
    ref_log = []

    # add the other ETF here so that the data for SPY will be validated against it, but we won't use it directly
    ref_port = MDPPortfolio(assets_list=['IWM', 'TLT'])
    ref_port = get_data(ref_port, base_etf=ref_port.assets[0], last_x_days=0, get_new_data=UPDATE_DATA, historical_data=historical_data)
    ref_price_list = ref_port.trimmed[ref_port.assets[0]]

    # first_date = datetime.datetime.strptime(mdp_port.portfolio_valuations[0][0], '%Y-%m-%d')
    # ref_closes = [t['AdjClose'] for t in ref_price_list if datetime.datetime.strptime(t['Date'], '%Y-%m-%d') >= first_date ]
    first_date = mdp_port.portfolio_valuations[0][0]
    ref_closes = [ t['AdjClose'] for t in ref_price_list if t['Date'] >= first_date ]
    for val in ref_closes:
        ref_log.append(('existing_trade', 'long', val))


    ### Gather statistics
    system_stats = {}
    ref_stats = {}

    system_stats['drawdown'] = get_drawdown(sys_closes)
    ref_stats['drawdown'] = get_drawdown(ref_closes)

    # smean, sstd, sneg_std, spos_std, ssharpe, ssortino, savg_loser, savg_winner, spct_losers = get_sharpe_ratio(sharpe_price_list)

    system_stats['arith_mean'], system_stats['sigma'], system_stats['neg_sigma'], system_stats['pos_sigma'], \
        system_stats['sharpe'], system_stats['sortino'], system_stats['avg_loser'], system_stats['avg_winner'], \
        system_stats['pct_losers'] = get_sharpe_ratio(sharpe_price_list)

    if write_to_file:
        output_fields = ('Date', 'Valuation', 'DR', 'Mean_Vol', 'Weighted_Mean_Vol', 'IsRebalanced')

        # We must add back in dashes into the date so Excel handles this properly
        output_string = '\n'.join( [(str(n[0][0:4]) + '-' + str(n[0][4:6]) + '-' + str(n[0][6:8])
                                     + ',' + str(n[1]) + ',' + str(ref_log[k][2]/ref_log[0][2]) +\
                                     ',' + str(system_stats['drawdown'][k]) + ',' + str(ref_stats['drawdown'][k]) +\
                                     ',' + str(system_dma[k]) +\
                                     ',' + str(mdp_port.trailing_DRs[k])
            ) for k, n in enumerate(mdp_port.portfolio_valuations)] )

        current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        out_file_name = '/home/wilmott/Desktop/fourseasons/fourseasons/results/portfolio_analysis' + '_' + str(current_time) +'.csv'
        with open(out_file_name, 'w') as f:
            f.write(output_string)

    system_stats['total_years_in'] = len(sharpe_price_list) / 252.0
    system_stats['annualized_return'] = math.pow(mdp_port.portfolio_valuations[-1][1], (1.0/system_stats['total_years_in']))

    # system_stats['mean_diversification_ratio'] = np.mean([ n for n in mdp_port.trailing_DRs if n != 0.0 ])
    historical_trailing_DRs = np.mean( [ n[1] for n in mdp_port.rebalance_log ])
    system_stats['mean_diversification_ratio'] = historical_trailing_DRs
    system_stats['number_of_rebals'] = len(mdp_port.rebalance_log)

    # We added leading zeroes to trailing_DRs, don't include them in the mean
    if logging.root.level < 25:
        print "Mean Diversification Ratio: ", system_stats['mean_diversification_ratio'], system_stats['number_of_rebals']
        print '\t\tSystem:'
        print 'ArithMu: \t', round(system_stats['arith_mean'], 6)
        print 'Sigma: \t\t', round(system_stats['sigma'], 6)
        print 'NegSigma: \t', round(system_stats['neg_sigma'], 6)
        print 'NegSigma/Tot: \t', round((system_stats['neg_sigma']/system_stats['sigma']), 6)
        print 'Sharpe: \t', round(system_stats['sharpe'], 6)
        print 'Sortino: \t', round(system_stats['sortino'], 6)
        print 'Ann. Return: \t', round(system_stats['annualized_return'], 6)

    ref_stats['arith_mean'], ref_stats['sigma'], ref_stats['neg_sigma'], ref_stats['pos_sigma'], \
        ref_stats['sharpe'], ref_stats['sortino'], ref_stats['avg_loser'], ref_stats['avg_winner'], \
        ref_stats['pct_losers'] = get_sharpe_ratio(ref_log)

    ref_total_return = ref_log[-1][2] / ref_log[0][2]
    ref_stats['annualized_return'] = math.pow(ref_total_return, (1.0/system_stats['total_years_in']))

    if logging.root.level < 25:
        print '\n\t\tReference:'
        print 'ArithMu: \t', round(ref_stats['arith_mean'], 6)
        print 'Sigma: \t\t', round(ref_stats['sigma'], 6)
        print 'NegSigma: \t', round(ref_stats['neg_sigma'], 6)
        print 'NegSigma/Tot: \t', round((ref_stats['neg_sigma']/ref_stats['sigma']), 6)
        print 'Sharpe: \t', round(ref_stats['sharpe'], 6)
        print 'Sortino: \t', round(ref_stats['sortino'], 6)
        print 'Ann. Return: \t', round(ref_stats['annualized_return'], 6)

        print "\nFinished: "
        if write_to_file:
            print "File Written: ", out_file_name.split('/')[-1]

    return system_stats

def get_port_valuation(port, x=0):
    total_delta = 0
    total_valuation = 0

    # print "\nENTRIES: ", port.current_entry_prices
    for k, v in enumerate(port.assets):
        # total_valuation += weights[k][0] * port.closes[v][x]
        if port.shares[k] >= 0:
            total_delta += port.shares[k] * (port.closes[v][x] - port.current_entry_prices[k])
            total_valuation += abs(port.shares[k] * port.closes[v][x])
        else:
            total_delta += abs(port.shares[k]) * (port.current_entry_prices[k] - port.closes[v][x])
            total_valuation += abs(port.shares[k]) * (port.current_entry_prices[k] + port.current_entry_prices[k] - port.closes[v][x])

    for k, v in enumerate(port.assets):
        # This loop is used to get current weights that are evaluated in real-time each day, rather than static
        # only every rebalance as the normalized weights are
        if port.shares[k] >= 0:
            this_delta = port.shares[k] * (port.closes[v][x] - port.current_entry_prices[k])
            this_weight = port.shares[k] * port.closes[v][x]
        else:
            this_delta = abs(port.shares[k]) * (port.current_entry_prices[k] - port.closes[v][x])
            this_weight = -abs(port.shares[k]) * (port.current_entry_prices[k] + port.current_entry_prices[k] - port.closes[v][x])

        # print "VALUATION: ", port.shares[k], port.closes[v][x], this_valuation, total_valuation, '\t', (this_valuation / total_valuation)
        port.current_weights[k] = this_weight / total_valuation

    logging.debug('%s %s %s' % (x, port.trimmed[port.assets[0]][x]['Date'], total_delta))

    return total_valuation

def get_drawdown(closes):

    all_time_high = closes[0]
    drawdown = [1.0]

    for k, close in enumerate(closes):
        if k == 0:
            continue
        all_time_high = max(close, all_time_high)
        drawdown.append(close / all_time_high)

    return drawdown

class MDPPortfolio():

    def __init__(self, assets_list):

        self.assets = assets_list
        self.normalized_weights = []
        self.portfolio_returns = []
        self.portfolio_valuations = []
        self.trailing_DRs = []

        # These should not get reset
        self.closes = {}
        self.trimmed = {}

        self.forward_returns = {}

        self.past_returns = {}
        self.volatilities = {}

        #Matrices / vectors
        self.past_returns_matrix = None
        self.volatilities_matrix = None
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.transposed_volatilities_matrix = None

        self.lookback = None
        self.rebalance_time = None

        for item in assets_list:
            self.closes[item] = []
            self.trimmed[item] = []
            self.past_returns[item] = []
            self.volatilities[item] = []

    def validate_portfolio(self):
        len_data = len(self.trimmed[self.assets[0]])
        start_date = self.trimmed[self.assets[0]][0]['Date']
        end_date = self.trimmed[self.assets[0]][-1]['Date']

        for item in self.assets:
            if len(self.trimmed[item]) != len_data:
                logging.info('Failed validation: %s %s %s %s' % (item, 'len trimmed', len(self.trimmed[item]), len_data))
                logging.info('Failed validation: %s %s %s %s' % (item, 'len closes', len(self.closes[item]), len_data))
                logging.info('Failed validation: %s %s %s %s' % (item, 'start dates', self.trimmed[item][0]['Date'], start_date))
                logging.info('Failed validation: %s %s %s %s' % (item, 'end dates', self.trimmed[item][-1]['Date'], end_date))
                return False
            if len(self.closes[item]) != len_data:
                logging.info('Failed validation: %s %s %s %s' % (item, 'len closes', len(self.closes[item]), len_data))
                return False

            if self.trimmed[item][0]['Date'] != start_date:
                logging.info('Failed validation: %s %s %s %s' % (item, 'start dates', self.trimmed[item][0]['Date'], start_date))
                return False
            if self.trimmed[item][-1]['Date'] != end_date:
                logging.info('Failed validation: %s %s %s %s' % (item, 'end_date', self.trimmed[item][-1]['Date'], end_date))
                return False

        return True

    def weighted_vols_equal_one(self, weights):
        normalized_weights = weights
        transposed_weights = np.matrix.transpose(normalized_weights)

        # logging.info('weighted_vols_equal_one: transposed_weights: %s' % (transposed_weights))
        # logging.info('weighted_vols_equal_one: volatilities_matrix: %s' % (self.volatilities_matrix))

        r1 = np.dot(transposed_weights, self.volatilities_matrix)
        # logging.info('Weighted Vols Value: %s' % (r1))

        # print transposed_weights, self.volatilities_matrix
        # print weights, '\n', self.volatilities_matrix
        print "#### R1: \n", r1

        return np.array([r1[0] - 1])

    def get_tangency_weights(self, x):
        end_index = x
        start_index = max(0, x - self.lookback)

        self.mean_past_returns = {}
        for item in self.assets:
            closes = self.closes[item][start_index:end_index]
            self.past_returns[item] = get_returns(closes)
            self.mean_past_returns[item] = np.mean(self.past_returns[item])
            self.volatilities[item] = np.std(self.past_returns[item])

        self.past_returns_matrix = np.array([self.past_returns[item] for item in self.assets])
        self.volatilities_matrix = np.array( [ [self.volatilities[z]] for z in self.assets ] )
        self.cov_matrix = np.cov(self.past_returns_matrix)

        # This is used as the "mean return"
        volatilities_matrix = self.volatilities_matrix
        self.transposed_volatilities_matrix = np.matrix.transpose(self.volatilities_matrix)

        unity_vector = np.array([[1]] * len(self.assets))

        # Either the unity vector OR the transposed volalities matrix can be used in B, below. Changing them would scale
        # the result differently, but we normalize as the last step. The white paper shows this calculation done with
        # the transposed volatilities matrix, but the website actually shows it done with the unity vector. Note that
        # the "B" that is used in the whitepaper is the "C" that is on the website.

        A = np.dot(np.dot(np.matrix.transpose(unity_vector), scipy.linalg.inv(self.cov_matrix)), unity_vector)
        # print "A is: ", A

        # B = np.dot(np.dot(np.matrix.transpose(unity_vector), scipy.linalg.inv(self.cov_matrix)), volatilities_matrix)
        B = np.dot(np.dot(self.transposed_volatilities_matrix, scipy.linalg.inv(self.cov_matrix)), volatilities_matrix)
        # print "B is: ", B

        C = np.dot(np.dot(self.transposed_volatilities_matrix, scipy.linalg.inv(self.cov_matrix)), volatilities_matrix)
        # print "C is: ", C

        #D = (A * C) - (B^2)
        #print "D is: ", D

        numerator = np.dot(scipy.linalg.inv(self.cov_matrix), volatilities_matrix)
        denominator = B
        tangency_weights = np.divide(numerator, denominator)

        # This is akin (though possibly not identical) to the "naive" risk parity optimization.
        ### tangency_weights = np.divide(unity_vector, self.volatilities_matrix)

        total_sum = sum([abs(n) for n in tangency_weights])[0]
        normalized_weights = np.divide(tangency_weights, total_sum)

        return normalized_weights


    def set_normalized_weights(self, weights, old_weighted_valuation, x):
        self.normalized_weights = weights
        weight_sum = np.sum([abs(n) for n in weights])
        if round(weight_sum, 6) != 1.0:
            raise Exception("set normalized weights don't sum to 1.0: " + str(weight_sum))
        self.current_entry_prices = [ self.closes[v][x] for k, v in enumerate(self.assets) ]
        self.shares = [ (old_weighted_valuation * self.normalized_weights[k][0] / self.closes[v][x]) for k, v in enumerate(self.assets) ]

    def get_diversification_ratio(self, weights=None):

        if weights == 'current':
            weights_to_use = self.current_weights
        elif weights == 'normalized':
            weights_to_use = self.normalized_weights

        # logging.info('\nGet Div Ratio: WEIGHTS: \n' + str(weights) + str('\n') + str(weights_to_use))

        transposed_weights = np.matrix.transpose(weights_to_use)
        numerator = np.dot(transposed_weights, self.volatilities_matrix)

        d1 = np.dot(transposed_weights, self.cov_matrix)
        d2 = np.dot(d1, weights_to_use)
        d3 = math.sqrt(d2)

        ret = (numerator / d3)[0][0]

        self.diversification_ratio = ret
        return ret

class TradeLog(object):

    def __init__(self):
        self.Date = None
        self.Valuation = None
        self.DR = None
        self.Mean_Vol = None
        self.Weighted_Mean_Vol = None
        self.IsRebalanced = None


def run_live_portfolio_analysis(assets=None):
    all_portfolios=assets if assets else live_portfolio

    for account in all_portfolios:
        cash = 0

        assets = [ n[0] for n in account ]
        shares = [ n[1] for n in account ]

        port = MDPPortfolio(assets)

        port = get_data(port, base_etf=port.assets[0], last_x_days=0, get_new_data=UPDATE_DATA)

        port.shares = shares
        port.current_entry_prices = [ 0 for n in port.assets ]
        port.current_weights = [ 0 for n in port.assets ]
        port.lookback = 63

        # print [ port.closes[v] for k, v in enumerate(port.assets) ]

        max_index = len(port.closes[port.assets[0]]) - 1

        tangency_weights = port.get_tangency_weights(x=max_index)

        # get_port_valuation() should be setting the current_weights
        valuation = get_port_valuation(port, x=max_index) + cash
        port.current_weights = np.array([ [n] for n in port.current_weights] )
        original_weights = port.current_weights
        div_ratio = port.get_diversification_ratio(weights='current')

        port.current_weights = tangency_weights
        possible_div_ratio = port.get_diversification_ratio(weights='current')

        print "Assets: \t", assets
        print "Tangency: \t", [ round(n[0], 6) for n in tangency_weights ]
        print "Act. Weights: \t", [ round(n[0], 6) for n in original_weights ]
        print "Value: \t\t", valuation
        print "Act Div Ratio: \t", div_ratio
        print "Tan Div Ratio: \t", possible_div_ratio

    return


    # for x in xrange(0, num_threads):
    #     process_list.append(multiprocessing.Process(target=worker_do_analysis, args=(queue, mgr_dict, mgr_list, lock, redis_historical_data_dict, sys.stdout)))

    # for x in xrange(0, num_threads):
    #     process_list[x].start()

    # p1.start()
    # p2.start()

    # queue.close()
    # queue.join_thread()

    # p1.join()
    # p2.join()
    # output_string += mgr_dict['output_string']





# def worker_do_analysis(q, mgr_dict, mgr_list, lock, redis_historical_data_dict, stream_out):
#     # while True:
#     # val = q.get()
#     while True:
#         try:
#             val = mgr_list.pop()
#         except:
#             return
#         system_stats = do_analysis(val, write_to_file=False, historical_data_dict=redis_historical_data_dict)
#         with lock:
#             mgr_dict['all_stats_list'].append(system_stats)
#             # "Items: ", k, ' of ', len(combinations), \
#             screen_output =   "\nItems Left: " + str(len(mgr_list)) + \
#                               " AnnRet: " + str(round(system_stats['annualized_return'], 6)) + \
#                               " Sigma: " + str(round(system_stats['sigma'], 6)) + \
#                               " Mu DR: " + str(round(system_stats['mean_diversification_ratio'], 6))
#                               # "Sharpe: ", round(system_stats['sharpe'], 6), \
#
#             stream_out.write(screen_output)
#
#             mgr_dict['output_string'] += str(round(system_stats['annualized_return'], 6)) + ','
#             mgr_dict['output_string'] += str(round(system_stats['sharpe'], 6)) + ','
#             mgr_dict['output_string'] += str(round(system_stats['sigma'], 6)) + ','
#             mgr_dict['output_string'] += str(round(system_stats['mean_diversification_ratio'], 6)) + ','
#             mgr_dict['output_string'] += ','.join([str(a) for a in val])
#             mgr_dict['output_string'] += '\n'
#         # q.task_done()