import numpy as np
import scipy as scipy
import datetime
import copy
import sys

import math
import itertools

import src.toolsx as tools

from src.indicator_system                           import get_sharpe_ratio
from src.portfolio_analysis.portfolio_utils         import get_data
from src.portfolio_analysis.portfolio_constants     import custom_assets_list, live_portfolio, stocks_to_test

from src.math_tools                                 import get_returns, get_ln_returns

import logging

# determine whether to download new data from the internet
UPDATE_DATA=True
logging.root.setLevel(logging.INFO)

def do_optimization(mdp_port, x):
    theoretical_weights = mdp_port.get_tangency_weights(x)

    # theoretical_weights = np.array([ [1.0 / len(mdp_port.assets)] for x in mdp_port.assets])
    # theoretical_weights = np.array([ [0.30], [0.15], [0.40], [0.075], [0.075] ])
    # theoretical_weights = np.array([ [0.30], [0.55], [0.15] ])

    # if logging.root.level < 25:
        # print "Theoretical Result: \n", theoretical_weights
        # print "weighted vols equal one check: ", mdp_port.weighted_vols_equal_one(theoretical_weights)

    # This section should not be necessary since weights are already normalized. It's an added safety.
    abs_weight_sum = np.sum([abs(n) for n in theoretical_weights])
    real_weight_sum = np.sum(theoretical_weights)
    weight_ratio = real_weight_sum / abs_weight_sum if round(abs_weight_sum, 6) != 1.0 else 1.0

    normalized_theoretical_weights = np.array([ n * weight_ratio for n in theoretical_weights])

    # Use this to equal weight all or a portion of the portfolio
    # equal_weighted_value = [1.0 / (len(theoretical_weights))]
    # normalized_theoretical_weights = np.array([ equal_weighted_value for n in theoretical_weights])

    # Turn this on to fix the weights of at each rebalance
    ### HACK:

    # SPY SECTORS: normalized_theoretical_weights = np.array([ [0.113], [0.09], [0.135], [0.0], [0.0], [0.215], [0.215], [0.0], [0.057], [0.175] ])
    # normalized_theoretical_weights = np.array([[0.20], [0.20], [0.20], [0.20], [0.20]])
    # normalized_theoretical_weights = np.array([[0.30], [0.15], [0.40], [0.075], [0.075]])
    normalized_theoretical_weights = np.array([[0.30], [0.15], [0.40], [0.075], [0.075], [0.0]])

    ##### ALGO WITH TREASURY DATA

    # must be <= 62
    ma_length = 10

    # import pdb; pdb.set_trace()

    recent_yc = [ d['AdjClose'] for d in mdp_port.other_data_trimmed[x-ma_length:x] ]
    yc_ma = np.mean(recent_yc)
    recent_yc_1 = [ d['AdjClose'] for d in mdp_port.other_data_trimmed[x-(ma_length+1):x-1] ]
    yc_ma_1 = np.mean(recent_yc_1)
    yc_ma_slope = yc_ma - yc_ma_1

    yc_ma_100 = 0.0
    if x > 100:
        yc_ma_100 = np.mean([ d['AdjClose'] for d in mdp_port.other_data_trimmed[x-100:x-80] ])

    adjusted_yc_ma = yc_ma - 0.0

    all_weather_base = [[0.30], [0.15], [0.40], [0.075], [0.075]]
    golden_butterfly_base = [[0.20], [0.20], [0.20], [0.20], [0.20]]

    # End of cycle, inflation area, use commodities
    if adjusted_yc_ma < 0 and yc_ma_slope < 0:
        normalized_theoretical_weights = np.array([[0.15], [0.15], [0.0], [0.85], [0.85], [-1.0]])
        # normalized_theoretical_weights = np.array([[0.15], [0.15], [0.20], [0.0], [1.5], [-1.0]])

    # bear market area, do not leverage, just stay in All Weather, but bias slightly to bonds and away from inflation
    elif adjusted_yc_ma < 0 and yc_ma_slope > 0:
        normalized_theoretical_weights = np.array([[0.10], [0.40], [0.40], [0.05], [0.05], [0.0]])
        # normalized_theoretical_weights = np.array([[0.05], [0.05], [0.40], [0.40], [0.10], [0.0]])

    # coming out of recession, stay in all weather, the normal rules don't apply, low commodities, heavy bonds
    elif adjusted_yc_ma >= 0 and yc_ma_slope > 0 and (yc_ma_100 - 1.0) < 0:
        normalized_theoretical_weights = np.array([[0.20], [0.30], [0.75], [0.05], [0.05], [-0.35]])
        # normalized_theoretical_weights = np.array([[0.10], [0.10], [0.35], [0.75], [0.05], [-0.35]])

    elif adjusted_yc_ma >=0 and adjusted_yc_ma < 1:
        weights = [ [1.00*n[0]] for n in all_weather_base]
        weights.append([0.0])
        normalized_theoretical_weights = np.array(weights)

    # just use all weather, the yields available are not high
    elif adjusted_yc_ma >= 1 and adjusted_yc_ma < 2:
        # normalized_theoretical_weights = np.array([[0.30], [0.15], [0.40], [0.075], [0.075]])
        weights = [ [3.50*n[0]] for n in all_weather_base]
        weights.append([-2.50])
        normalized_theoretical_weights = np.array(weights)

    # 2x leverage, go into risk assets
    elif adjusted_yc_ma >= 2.0 and adjusted_yc_ma < 3.0:
        # normalized_theoretical_weights = np.array([[1.3], [0.15], [0.40], [0.075], [0.075]])
        weights = [ [4.0*n[0]] for n in all_weather_base]
        weights.append([-3.00])
        normalized_theoretical_weights = np.array(weights)

    # 3x leverage, go higher into risk assets
    elif adjusted_yc_ma >= 3.0 and adjusted_yc_ma < 4.0:
        # normalized_theoretical_weights = np.array([[2.3], [0.15], [0.40], [0.075], [0.075]])
        weights = [ [5.00*n[0]] for n in all_weather_base]
        weights.append([-4.00])
        normalized_theoretical_weights = np.array(weights)

    elif adjusted_yc_ma >= 4.0 and adjusted_yc_ma < 5.0:
        # normalized_theoretical_weights = np.array([[3.3], [0.15], [0.40], [0.075], [0.075]])
        weights = [ [5.0*n[0]] for n in all_weather_base]
        weights.append([-4.0])
        normalized_theoretical_weights = np.array(weights)


    # normalized_theoretical_weights = np.array( [[0.20], [0.20], [0.20], [0.20], [0.20], [0.0]])
    # normalized_theoretical_weights = np.array([ [1*n[0]] for n in all_weather_base ])
    # normalized_theoretical_weights = np.array([[0.90], [0.45], [1.20], [0.225], [0.225], [-2.0]])


    print 'Adjusting Weights: ', x, mdp_port.other_data_trimmed[x]['Date'], yc_ma_slope, yc_ma_100, '\n', normalized_theoretical_weights
    # import pdb; pdb.set_trace()



    # Rising Inflation, Growth and Flattening, Positive YC - Commodities

    # Falling Growth, High Inflation And Negative And Falling YC - Cash

    # Low Growth, Low Inflation, Rising, Negative YC, Or Steeply Rising and Positive YC - Bonds

    # Higher Growth, Low Inflation, Positive YC That is not Changing Slope Too Quickly - Stocks

    # Fa


    #####

    # normalized_theoretical_weights = np.array([[0.15], [0.15], [0.20], [0.075], [0.075], [0.15], [0.20]])
    # normalized_theoretical_weights = np.array([[0.20], [0.15], [0.20], [0.15], [0.10], [0.20]])
    # normalized_theoretical_weights = np.array([[0.35], [0.18], [0.47]])

    # Modified All Weather 1 rebalance for all of history, done on 20161120
    # normalized_theoretical_weights = np.array([[0.15], [0.19], [0.28], [0.08], [0.09], [0.09], [0.12]])

    # modified_mdp optimized for 2008 days, single rebalance
    # normalized_theoretical_weights = np.array([[0.23], [0.41], [0.17], [0.09], [0.10]])

    # MDP Optimized For 2338/2339 historical days single rebalance
    # normalized_theoretical_weights = np.array([[0.25], [0.28], [0.29], [0.08], [0.10]])

    # Naive Risk Parity Optimized for 2338/2339 historical days single rebalance
    # normalized_theoretical_weights = np.array([[0.14], [0.40], [0.19], [0.13], [0.14]])

    return mdp_port, normalized_theoretical_weights

def run_portfolio_analysis():
    logging.root.setLevel(logging.INFO)
    do_analysis(assets=custom_assets_list, write_to_file=True)
    return

def do_analysis(assets=None, write_to_file=True):

    assets_list = assets

    mdp_port = MDPPortfolio(assets_list=assets_list)
    logging.debug(str(mdp_port.assets))
    mdp_port = get_data(mdp_port, base_etf=mdp_port.assets[0], last_x_days=0, get_new_data=UPDATE_DATA)

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
    old_weighted_valuation = 1000
    debt = 0
    while x < len(mdp_port.trimmed[mdp_port.assets[0]]):
        if not mdp_port.rebalance_now:

            mdp_port.x = x
            ### current_portfolio_valuation = get_port_valuation(mdp_port, x=x)
            ### current_leverage_ratio = np.sum(mdp_port.normalized_weights)
            ### current_portfolio_valuation = get_port_valuation(mdp_port, x=x) - ((current_leverage_ratio - 1) * old_weighted_valuation)

            current_portfolio_valuation = get_port_valuation(mdp_port, x=x) - debt
            current_leverage_ratio = current_portfolio_valuation / (current_portfolio_valuation - debt)

            ### print "Not Rebalancing: ", current_portfolio_valuation, get_port_valuation(mdp_port, x=x), old_weighted_valuation, current_leverage_ratio, ((current_leverage_ratio - 1) * old_weighted_valuation)
            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation])
            # if logging.root.level < 25:
                ### print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation
            mdp_port.rebalance_counter += 1

            if x >= mdp_port.rebalance_time:
                # The statement below DOES have an impact on whether a rebalance is hit in this if block
                # It also affects overall program flow and the end result
                _ = mdp_port.get_tangency_weights(x)
                trailing_diversification_ratio = mdp_port.get_diversification_ratio(weights='current')
                mdp_port.trailing_DRs.append(trailing_diversification_ratio)
                rebalance_date = mdp_port.trimmed[mdp_port.assets[0]][x]['Date']
                # if logging.root.level < 25:
                    ### print "Trailing DR: ", x, trailing_diversification_ratio

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

            ### old_leverage_ratio = np.sum(mdp_port.normalized_weights)
            old_weighted_valuation = get_port_valuation(mdp_port, x=x) - debt
            mdp_port.x = x

            # old_weights = mdp_port.normalized_weights

            rebalance_old_div_ratio = mdp_port.get_diversification_ratio(weights='current')

            mdp_port, theoretical_weights = do_optimization(mdp_port, x)

            ###
            # if logging.root.level < 25:
            #     print "\n\n\nTheoretical Weights: ", str(theoretical_weights)
            mdp_port.set_normalized_weights(theoretical_weights, old_weighted_valuation, x)
            ###

            # calling get_port_valuation here is used to set the current weights, used in the div ratio below
            _ = get_port_valuation(mdp_port, x=x)
            ### for k, asset in enumerate(mdp_port.assets):
            ###     if round(mdp_port.normalized_weights[k], 6) != round(mdp_port.current_weights[k], 6):
            ###        raise Exception("Normalized weights do not equal the current weights after setting normalized.")

            trailing_diversification_ratio = mdp_port.get_diversification_ratio(weights='normalized')
            mdp_port.trailing_DRs.append(trailing_diversification_ratio)
            if logging.root.level < 25:
                print "New Trailing Diversification Ratio: ", x, trailing_diversification_ratio
            mdp_port.rebalance_log.append((rebalance_date, rebalance_old_div_ratio, trailing_diversification_ratio, theoretical_weights))

            # total_leverage = np.sum(old_weights)
            current_leverage_ratio = np.sum(mdp_port.normalized_weights)
            debt = (current_leverage_ratio - 1) * old_weighted_valuation
            current_portfolio_valuation = get_port_valuation(mdp_port, x=x) - debt

            ### current_portfolio_valuation = get_port_valuation(mdp_port, x=x) / current_leverage_ratio
            ### if round(current_portfolio_valuation, 6) != round(old_weighted_valuation, 6):
            ###    raise Exception("Current Valuation != Old Weighted Valuation: Curr/Old: " + str(current_portfolio_valuation) + '\t' + str(old_weighted_valuation))

            # if logging.root.level < 25:
            #     print "Current Portfolio New Valuation: ", current_portfolio_valuation

            print "Appending Valuation: ", mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_leverage_ratio, current_portfolio_valuation
            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation ])
            # if logging.root.level < 25:
            #     print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation
            x+=1

    for i, p in enumerate(mdp_port.portfolio_valuations):
        if i == 0:
            starting_value = p[1]
        mdp_port.portfolio_valuations[i][1] = mdp_port.portfolio_valuations[i][1] / starting_value

    if len(mdp_port.trimmed[mdp_port.assets[0]]) != len(mdp_port.portfolio_valuations):
        raise Exception('Length of asset list data does not match length of valuations data.')

    system_results = aggregate_statistics(mdp_port, write_to_file=write_to_file)
    return system_results


def aggregate_statistics(mdp_port, write_to_file=True):

    if logging.root.level < 25:
        print '\n'.join([str(r) for r in mdp_port.rebalance_log])

    ### Create sharpe_price_list and a Moving Average of port valuations. Last if statement will simply store days
    # the valuation was < MA
    sharpe_price_list = []
    sys_closes = []
    system_dma = []
    # trade_days_skipped = []
    for k, val in enumerate(mdp_port.portfolio_valuations):
        sharpe_price_list.append(('existing_trade', 'long', val[1]))
        sys_closes.append(val[1])
        if k < 200:
            system_dma.append(1)
        else:
            system_dma.append(np.mean([n[1] for n in mdp_port.portfolio_valuations[k-200:k+1]]))
        # if mdp_port.portfolio_valuations[k-1][1] < system_dma[k-1]:
        #     trade_days_skipped.append(k)

    ### Show starting and ending valuation (after potentially adding a moving average filter)
    if logging.root.level < 25:
        print '\n', mdp_port.portfolio_valuations[0], mdp_port.portfolio_valuations[-1]

    ### Get Ref Data
    ref_log = []

    # add the other ETF here so that the data for SPY will be validated against it, but we won't use it directly
    ref_port = MDPPortfolio(assets_list=['SPY', 'TLT'])
    ref_port = get_data(ref_port, base_etf=ref_port.assets[0], last_x_days=0, get_new_data=UPDATE_DATA)
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
        output_fields = ('Date', 'Port Valuation', 'Ref Valuation', 'Port DD', 'Ref DD', 'Port MA', 'Port DRs')

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

    system_stats['mean_diversification_ratio'] = np.mean([ n for n in mdp_port.trailing_DRs if n != 0.0 ])
    #historical_trailing_DRs = np.mean( [ n[1] for n in mdp_port.rebalance_log ])
    #system_stats['mean_diversification_ratio'] = historical_trailing_DRs
    system_stats['number_of_rebals'] = len(mdp_port.rebalance_log)

    # We added leading zeroes to trailing_DRs, don't include them in the mean
    if logging.root.level < 25:
        print "Mean Daily Diversification Ratio: ", system_stats['mean_diversification_ratio'], len(mdp_port.trailing_DRs)
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
    total_long_valuation = 0
    total_short_valuation = 0
    short_delta = 0

    # print "\nENTRIES: ", port.current_entry_prices
    for k, v in enumerate(port.assets):
        # total_valuation += weights[k][0] * port.closes[v][x]
        if port.shares[k] >= 0:
            total_delta += port.shares[k] * (port.closes[v][x] - port.current_entry_prices[k])
            total_valuation += abs(port.shares[k] * port.closes[v][x])
            total_long_valuation += abs(port.shares[k] * port.closes[v][x])
        ### HACK - this forces us to treat shorting cash as a special case that does not add to risk at all
        # Just ignore anything with negative shares...rather than viewing this as a "short" position, that is adding to
        # to your assets at risk, we will instead just treat it as shorting cash, representing borrowed money
        else:
            total_delta += abs(port.shares[k]) * (port.current_entry_prices[k] - port.closes[v][x])
            total_valuation += abs(port.shares[k]) * (port.current_entry_prices[k] + port.current_entry_prices[k] - port.closes[v][x])
            total_short_valuation += abs(port.shares[k]) * (port.current_entry_prices[k] + port.current_entry_prices[k] - port.closes[v][x])

    for k, v in enumerate(port.assets):
        # This loop is used to get current weights that are evaluated in real-time each day, rather than static
        # only every rebalance as the normalized weights are
        if port.shares[k] >= 0:
            this_delta = port.shares[k] * (port.closes[v][x] - port.current_entry_prices[k])
            this_weight = port.shares[k] * port.closes[v][x]
        else:
            this_delta = abs(port.shares[k]) * (port.current_entry_prices[k] - port.closes[v][x])
            this_weight = -abs(port.shares[k]) * (port.current_entry_prices[k] + port.current_entry_prices[k] - port.closes[v][x])
            short_delta = abs(port.shares[k]) * (port.current_entry_prices[k] - port.closes[v][x])

        # print "VALUATION: ", port.shares[k], port.closes[v][x], this_valuation, total_valuation, '\t', (this_valuation / total_valuation)
        port.current_weights[k] = this_weight / total_valuation

    logging.debug('%s %s %s' % (x, port.trimmed[port.assets[0]][x]['Date'], total_delta))

    # The short cash concept introduces a parasitic loss each period. Here we must deduct it. Cannot look at its value
    # directly or sum it with a total valuation.
    net_long_valuation = total_long_valuation - total_short_valuation + short_delta

    # print "Valuation: ", total_valuation, net_long_valuation, total_short_valuation, port.current_entry_prices[-1], port.closes['BSV'][x], short_delta

    return net_long_valuation


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

        # Use to store signals, etc.
        self.use_other_data = True
        self.other_data = '10_year_treasury_minus_fed_funds_rate'
        self.other_data_trimmed = []

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
        ### if round(weight_sum, 6) != 1.0:
        ###     raise Exception("set normalized weights don't sum to 1.0: " + str(weight_sum))
        self.current_entry_prices = [ self.closes[v][x] for k, v in enumerate(self.assets) ]
        self.shares = [ (old_weighted_valuation * self.normalized_weights[k][0] / self.closes[v][x]) for k, v in enumerate(self.assets) ]
        print "SELF SHARES: ", self.shares

    def get_diversification_ratio(self, weights=None):

        # Current weights are the weights at a given instant, normalized weights are the weights at the last rebalance
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

    update_data = True
    update_date = '20180102'

    input = live_portfolio[0] + stocks_to_test

    port_ret = live_portfolio_analysis(assets=live_portfolio, update_data=update_data, update_date=update_date)
    port_prices_only = [ n[2] for n in port_ret['sharpe_price_list'] ]
    port_pct_rets_only = [0.0]
    for k, x in enumerate(port_prices_only):
        if k > 0:
            port_pct_rets_only.append( ( x -  port_prices_only[k-1]) / port_prices_only[k-1] )

    print '\n\n'

    ###
    # return


    output_data = []
    for stock_tuple in input:
        try:
            stock_ret = live_portfolio_analysis(assets=[[stock_tuple]], update_data=update_data, update_date=update_date)
            # None will be returned if this failed...usually as a result of data not found in redis for the input list
            if stock_ret is None:
                continue

            stock_prices_only = [ n[2] for n in stock_ret['sharpe_price_list'] ]
            stock_pct_rets_only = [0.0]
            for k, x in enumerate(stock_prices_only):
                if k > 0:
                    stock_pct_rets_only.append( ( x -  stock_prices_only[k-1]) / stock_prices_only[k-1] )


            # cov_matrix_input = np.array( [stock_pct_rets_only, port_pct_rets_only] )
            cov_matrix_input = np.array( [stock_pct_rets_only, port_pct_rets_only] )

            cov_matrix = np.cov(cov_matrix_input)
            variance = np.var(port_pct_rets_only)
            beta = cov_matrix[0][1] / variance
            sigma = np.sqrt(np.var(stock_pct_rets_only))
            output_data.append( (stock_tuple[0], beta, sigma) )

            # print "Beta: ", stock_tuple[0], '\t', 100.0 * beta, '\t\t', np.sqrt(np.var(stock_pct_rets_only))
        except:
            continue

    print '\n\n'
    print '     ', 'Symbol', '\t', 'Beta', '\t\t', 'Sigma'

    new_stock_output_data = output_data[len(live_portfolio[0]):]

    print '\n'
    # for item in sorted(output_data[0:len(live_portfolio[0])], key=lambda tup: tup[1], reverse=False):
        # print "Beta: ", item[0], '\t', item[1], '\t\t', item[2]
    for i, item in enumerate(output_data[0:len(live_portfolio[0])]):
        weighted_delta = (port_ret['tangency'][i] - port_ret['actual_weights'][i]) * port_ret['valuation']
        print "Beta: ", item[0], '\t', '{0:.6f}'.format(item[1].round(6)).zfill(8), '\t', '{0:.6f}'.format(item[2].round(6)).zfill(8), '\tAct / Tan / Diff:  ', port_ret['actual_weights'][i], '\t', port_ret['tangency'][i], '\t', weighted_delta
    for item in sorted(new_stock_output_data, key=lambda tup: tup[1], reverse=False):
        print "Beta: ", item[0], '\t', '{0:.6f}'.format(item[1].round(6)), '\t', '{0:.6f}'.format(item[2].round(6))

    return


def live_portfolio_analysis(assets=None, update_data=True, update_date=None):
    all_portfolios=assets if assets else live_portfolio

    for account in all_portfolios:
        cash = 0

        assets = [ n[0] for n in account ]
        shares = [ n[1] for n in account ]

        port = MDPPortfolio(assets)

        try:
            port = get_data(port, base_etf=port.assets[0], last_x_days=64, get_new_data=update_data, update_date=update_date)
        except Exception as e:
            return None

        port.shares = shares
        port.current_entry_prices = [ 0 for n in port.assets ]
        port.current_weights = [ 0 for n in port.assets ]
        port.lookback = 63

        # print [ port.closes[v] for k, v in enumerate(port.assets) ]

        max_index = len(port.closes[port.assets[0]]) - 1

        # We can ONLY do this if we have multiple assets to re-weight, otherwise, the weight is always fixed at 1.0.
        # This would be the case if we are only interested in the returns stats of a given asset
        if len(assets) > 1:
            tangency_weights = port.get_tangency_weights(x=max_index)
        else:
            tangency_weights = [ [1.0] ]

        historical_valuations = []
        sharpe_price_list = []
        for x in xrange(max_index-63, max_index):
            valuation = get_port_valuation(port, x=x)
            historical_valuations.append(valuation)
            sharpe_price_list.append(('existing_trade', 'long', valuation))
        a, system_sigma, c, d, e, f, g, h, i = get_sharpe_ratio(sharpe_price_list)
        # get_port_valuation() should be setting the current_weights
        valuation = get_port_valuation(port, x=max_index) + cash
        port.current_weights = np.array([ [n] for n in port.current_weights] )
        original_weights = port.current_weights

        if len(assets) > 1:
            # First calc of div ratio just uses the weights as-is, based on # of shares and prices
            div_ratio = port.get_diversification_ratio(weights='current')

            # Set the weights to the tangency weights, then recalc the div ratio
            port.current_weights = tangency_weights
            possible_div_ratio = port.get_diversification_ratio(weights='current')
        else:
            div_ratio = 1.0
            tangency_weights = [ [1.0] ]
            possible_div_ratio = 1.0




        if len(assets) > 1:
            print port.portfolio_valuations
            print "Assets: \t", assets
            print "Act. Sigma: \t", round(system_sigma, 6)
            print "Act. Weights: \t", [ str(round(n[0], 4)).zfill(6) for n in original_weights ]
            print "Tangency: \t", [ str(round(n[0], 4)).zfill(6) for n in tangency_weights ]
            print "Value: \t\t", valuation
            print "Act Div Ratio: \t", div_ratio
            print "Tan Div Ratio: \t", possible_div_ratio

        ret_dict = {'assets': assets, 'actual_sigma': round(system_sigma, 6), \
                'actual_weights': [ round(n[0], 6) for n in original_weights ], \
                'tangency': [ round(n[0], 6) for n in tangency_weights ],
                'valuation': valuation, 'actual_div_ratio': div_ratio, \
                'tangency_div_ratio': possible_div_ratio, \
                'sharpe_price_list': sharpe_price_list}

        return ret_dict



