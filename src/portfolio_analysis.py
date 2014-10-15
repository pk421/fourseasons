import numpy as np
import scipy as scipy
import datetime
import time
import copy

import math

import toolsx as tools
from data.redis import manage_redis
from src.data_retriever import load_redis
from src.data_retriever import multithread_yahoo_download
from src.cointegrations_data import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly, \
                                    get_bunches_of_pairs

from src.indicator_system       import get_sharpe_ratio

from math_tools import get_returns

import logging
logging.root.setLevel(logging.INFO)

USE_THEORETICAL = 'theoretical'
USE_BRUTE = False


def run_portfolio_analysis():

    # Brute Force Calcs = # nodes ^ # assets
    # assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
    # assets_list = ['TNA', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'SLV']
    # assets_list = ['VTI', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
    # assets_list = ['SPXL', 'TYD', 'DRN', 'DGP', 'EDC']

    # Setting IEF simulates a 3x fund, use TYD to actually get 3x, TLT has more data and TMF (3x) is more liquid, it works well
    # assets_list = ['VTI', 'TLT', 'VNQ', 'VWO', 'GLD']
    # assets_list = ['VTI', 'TMF', 'VNQ', 'VWO', 'GLD']
    # assets_list = ['VTI', 'TYD', 'DRN', 'VWO', 'DGP'] # Leveraged Version
    assets_list = ['IWM', 'TLT', 'VNQ', 'VWO', 'GLD']

    # Mom's:
    # assets_list = ['VTI', 'IEF', 'REK', 'VWO']

    # TSP: G | C, F, S, I - (consider the G fund cash since it can't go down)
    # assets_list = ['SPY', 'AGG', 'FSEMX', 'EFA']

    # IRA: Fidelity Commission Free:
    # S&P 500, US Small Cap, Short Term Treasury Bonds, Total US Bond Market, Dow Jones Real Estate, EAFE, BRICS, Emerging Markets, Gold Miners
    # assets_list = ['IVV', 'IJR', 'SHY', 'AGG', 'IYR', 'IEFA', 'BKF', 'IEMG', 'RING']
    # assets_list = ['IYR', 'IEFA', 'IEMG']
    # assets_list = ['VTI', 'IEF', 'VNQ', 'EEM']

    # 401k
    # Note: VBMPX has a shorter duration and is actually less volatile than VIPIX
    # VEMPX is small caps basically but has not done as well in this portfolio compared to VIIIX
    # The five asset list here is less volatile and better diversified, but has a lower return
    # assets_list = ['VIIIX', 'VEMPX', 'VTPSX', 'VIPIX', 'VBMPX']
    # assets_list = ['VIIIX', 'VEU', 'LAG']
    # assets_list = ['VIIIX', 'VTPSX', 'VIPIX']

    # IRA + 401k + Outside Invs
    # assets_list = ['IYR', 'IEFA', 'IEMG', 'VIIIX', 'VTPSX', 'VIPIX', 'AGG', 'GLD']


    # assets_list = ['VTI', 'VGK', 'VWO', 'GLD', 'VNQ', 'TIP', 'TLT', 'AGG', 'LQD']

#    in_file_name = 'list_dow_modified'
#    location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + in_file_name + '.csv'
#    in_file = open(location, 'r')
#    stock_list = in_file.read().split('\n')
#    for k, item in enumerate(stock_list):
#        new_val = item.split('\r')[0]
#        stock_list[k] = new_val
#    in_file.close()

    mdp_port = MDPPortfolio(assets_list=assets_list)
    logging.debug(str(mdp_port.assets))
    mdp_port = get_data(mdp_port, base_etf=mdp_port.assets[0], last_x_days=0)

    mdp_port.weights = [ [1.0 / len(mdp_port.assets)] for x in mdp_port.assets]
     ### mdp_port.weights = [ [0.72], [0.14], [0.14] ]

    # Start with 1000 dollars and buy appropriate number of shares in each item
    mdp_port.starting_valuation = 1000
    mdp_port.shares = [ (mdp_port.starting_valuation * mdp_port.weights[k][0] / mdp_port.closes[v][0]) for k, v in enumerate(mdp_port.assets) ]
    mdp_port.current_entry_prices = [ mdp_port.closes[v][0] for k, v in enumerate(mdp_port.assets) ]
    mdp_port.weights = np.array(mdp_port.weights)
    mdp_port.normalized_weights = mdp_port.weights
    mdp_port.current_weights = mdp_port.weights

    x=0
    mdp_port.x = x
    mdp_port.lookback = 63
    mdp_port.rebalance_time = 63
    mdp_port.rebalance_counter = 0
    mdp_port.rebalance_now = False
    previous_rebalance = 1
    # The previous rebalance remains necessary because it is only set when a rebalance is done. The way that shorts are
    # handled in the valuation function is that they are subtracted from the portfolio. A new valuation could end up
    # subtracting a large dollar value of shorts from the portfolio after revaluing, unless the portfolio is normalized
    # first.

    rebalance_log = []
    while x < len(mdp_port.trimmed[mdp_port.assets[0]]):
        if not mdp_port.rebalance_now:

            mdp_port.x = x
            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)
            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance])
            print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance
            mdp_port.rebalance_counter += 1

            if x >= mdp_port.rebalance_time:
                # The statement below DOES have an impact on whether a rebalance is hit in this if block
                # It also affects overall program flow and the end result
                _ = mdp_port.get_covariance_matrix(x)
                trailing_diversification_ratio = mdp_port.get_diversification_ratio()
                mdp_port.trailing_DRs.append(trailing_diversification_ratio)
                rebalance_date = mdp_port.trimmed[mdp_port.assets[0]][x]['Date']
                rebalance_old_div_ratio = trailing_diversification_ratio
                print "Trailing DR: ", x, trailing_diversification_ratio

                # Only consider a rebalance after rebalance_time, if div ratio is low, rebalance. If it's high, wait
                # another rebalance_time to check again
                if mdp_port.rebalance_counter >= mdp_port.rebalance_time and trailing_diversification_ratio < 10.0:
                    mdp_port.rebalance_now = True
                elif mdp_port.rebalance_counter >= mdp_port.rebalance_time and trailing_diversification_ratio >= 10.0:
                    mdp_port.rebalance_now = False
                    # mdp_port.rebalance_counter = 0
            x += 1
            continue
        else:
            mdp_port.rebalance_counter = 0
            mdp_port.rebalance_now = False
            # When calculating the rebalance ratio, we first assume we used the old weights for today's valuation. Then
            # we calculate new weights for today, value the portfolio for today, then find the ratio for today if we
            # had used the old weights for today
            old_weighted_valuation = get_port_valuation(mdp_port, x=x) / previous_rebalance
            # This is the portfolio value today, IF we had been using the previous weighting
            print "Today's value @ old weighting: ", old_weighted_valuation
            mdp_port.x = x

            mdp_port, normalized_weights, theoretical_weights = do_optimization(mdp_port, x)

            trailing_diversification_ratio = mdp_port.get_diversification_ratio()
            mdp_port.trailing_DRs.append(trailing_diversification_ratio)
            print "New Trailing Diversification Ratio: ", x, trailing_diversification_ratio

            # rebalance_log.append((rebalance_date, rebalance_old_div_ratio, trailing_diversification_ratio))
            rebalance_log.append((rebalance_date, rebalance_old_div_ratio, trailing_diversification_ratio, mdp_port.normalized_weights))

            print "Constrained (+) Result: \n", normalized_weights
            print "weighted vols equal one check: ", mdp_port.weighted_vols_equal_one(normalized_weights)

            ###
            if USE_THEORETICAL == 'theoretical':
                mdp_port.set_normalized_weights(theoretical_weights, x)
            elif USE_THEORETICAL == 'optimized':
                mdp_port.set_normalized_weights(normalized_weights, x)
            # In this case, use the theoretical iff they are all positive and sum to 1
            elif USE_THEORETICAL == 'hybrid':
                negative_weights = [ w for w in theoretical_weights if w < 0 ]
                if len(negative_weights) == 0:
                    mdp_port.set_normalized_weights(theoretical_weights, x)
                else:
                    mdp_port.set_normalized_weights(normalized_weights, x)
            ###

            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)
            mdp_port.starting_valuation = current_portfolio_valuation
            print "Current Portfolio New Valuation: ", current_portfolio_valuation
            previous_rebalance = current_portfolio_valuation / old_weighted_valuation
            print "Previous Rebalance: ", previous_rebalance

            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance])
            print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance
            x+=1

    print '\n'.join([str(r) for r in rebalance_log])
    
    for i, p in enumerate(mdp_port.portfolio_valuations):
        if i == 0:
            starting_value = p[1]
        mdp_port.portfolio_valuations[i][1] = mdp_port.portfolio_valuations[i][1] / starting_value


    if len(mdp_port.trimmed[mdp_port.assets[0]]) != len(mdp_port.portfolio_valuations):
        raise Exception('Length of asset list data does not match length of valuations data.')

    # print mdp_port.trimmed[mdp_port.assets[0]][0], mdp_port.trimmed[mdp_port.assets[0]][-1]
    print '\n', mdp_port.portfolio_valuations[0], mdp_port.portfolio_valuations[-1]

    sharpe_price_list = []
    sys_closes = []
    for val in mdp_port.portfolio_valuations:
        sharpe_price_list.append(('existing_trade', 'long', val[1]))
        sys_closes.append(val[1])

    ref_log = []
    ref_price_list = mdp_port.trimmed[mdp_port.assets[0]]
    first_date = datetime.datetime.strptime(mdp_port.portfolio_valuations[0][0], '%Y-%m-%d')
    ref_closes = [t['AdjClose'] for t in ref_price_list if datetime.datetime.strptime(t['Date'], '%Y-%m-%d') >= first_date ]
    for val in ref_closes:
        ref_log.append(('existing_trade', 'long', val))

    system_drawdown = get_drawdown(sys_closes)
    ref_drawdown = get_drawdown(ref_closes)

    smean, sstd, sneg_std, spos_std, ssharpe, ssortino, savg_loser, savg_winner, spct_losers = get_sharpe_ratio(sharpe_price_list)

    output_fields = ('Date', 'Valuation', 'DR', 'Mean_Vol', 'Weighted_Mean_Vol', 'IsRebalanced')

    output_string = '\n'.join( [(str(n[0]) + ',' + str(n[1]) + ',' + str(ref_log[k][2]/ref_log[0][2]) +
                                 ',' + str(system_drawdown[k]) + ',' + str(ref_drawdown[k])) for k, n in enumerate(mdp_port.portfolio_valuations)] )
    current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_file_name = '/home/wilmott/Desktop/fourseasons/fourseasons/results/portfolio_analysis' + '_' + str(current_time) +'.csv'
    with open(out_file_name, 'w') as f:
        f.write(output_string)

    total_years_in = len(sharpe_price_list) / 252.0
    system_annualized_return = math.pow(mdp_port.portfolio_valuations[-1][1], (1.0/total_years_in))

    print "Mean Diversification Ratio: ", np.mean(mdp_port.trailing_DRs), len(rebalance_log)
    print '\t\tSystem:'
    print 'ArithMu: \t', round(smean, 6)
    print 'Sigma: \t\t', round(sstd, 6)
    print 'NegSigma: \t', round(sneg_std, 6)
    print 'NegSigma/Tot: \t', round((sneg_std/sstd), 6)
    print 'Sharpe: \t', round(ssharpe, 6)
    print 'Sortino: \t', round(ssortino, 6)
    print 'Ann. Return: \t', round(system_annualized_return, 6)

    rmean, rstd, rneg_std, rpos_std, rsharpe, rsortino, ravg_loser, ravg_winner, rpct_losers = get_sharpe_ratio(ref_log)

    ref_total_return = ref_log[-1][2] / ref_log[0][2]
    ref_annualized_return = math.pow(ref_total_return, (1.0/total_years_in))

    print '\n\t\tReference:'
    print 'ArithMu: \t', round(rmean, 6)
    print 'Sigma: \t\t', round(rstd, 6)
    print 'NegSigma: \t', round(rneg_std, 6)
    print 'NegSigma/Tot: \t', round((rneg_std/rstd), 6)
    print 'Sharpe: \t', round(rsharpe, 6)
    print 'Sortino: \t', round(rsortino, 6)
    print 'Ann. Return: \t', round(ref_annualized_return, 6)


    print "\nFinished: "
    print "File Written: ", out_file_name.split('/')[-1]

    return

def do_optimization(mdp_port, x):

#   result = scipy.optimize.minimize(mdp_port.optimize, mdp_port.weights, jac=mdp_port.optimize_jacobian, method='SLSQP', options={'xtol': 1e-8, 'disp': True}, bounds = [(0,1) for z in mdp_port.assets] , constraints=port_constraints)

    nw = mdp_port.normalized_weights
    # This updates the covariance matrix using the current volatilities, so we can optimize based on that
    theoretical_weights = mdp_port.get_covariance_matrix(x)
    cm = mdp_port.cov_matrix
    assets = mdp_port.assets

    if USE_BRUTE:
        result = scipy.optimize.brute(optimize, [(0,1) for z in assets], (cm,), Ns=21, full_output=False)
        sum_result = sum(result)
        normalized_weights = np.array([[round(z / sum_result, 4)] for z in result])
        print result[0], sum_result


    else:
        mk = {'args':(cm,), "method":"L-BFGS-B"}
        

        # myBounds = MyBounds()
        result = scipy.optimize.basinhopping(optimize, nw, niter=2, disp=True, minimizer_kwargs=mk)
        # result = scipy.optimize.basinhopping(optimize, nw, niter=1000, disp=True, minimizer_kwargs=mk, accept_test=myBounds)
        # result = scipy.optimize.basinhopping(optimize, nw, niter=50, T=5e-2, stepsize=5e-2, disp=True, minimizer_kwargs=mk)

        sum_result = sum(result.x)
        normalized_weights = np.array([[round(z / sum_result, 4)] for z in result.x])

    print result

    print "Theoretical Result: \n", theoretical_weights
    print "weighted vols equal one check: ", mdp_port.weighted_vols_equal_one(mdp_port.normalized_weights)


    return mdp_port, normalized_weights, theoretical_weights

class MyBounds(object):
    def __init__(self, xmax=[1.0], xmin=[-0.5]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin



def get_port_valuation(port, x=0):
    ### TODO: accept a parameter for rebalance ratio so that it is not divided everywhere else

    total_delta = 0
    total_valuation = 0

    # print "\nENTRIES: ", port.current_entry_prices
    for k, v in enumerate(port.assets):
        # total_valuation += weights[k][0] * port.closes[v][x]
        total_delta += port.shares[k] * (port.closes[v][x] - port.current_entry_prices[k])
        total_valuation += abs(port.shares[k] * port.closes[v][x])

    for k, v in enumerate(port.assets):
        this_delta = port.shares[k] * (port.closes[v][x] - port.current_entry_prices[k])
        this_valuation = abs(port.shares[k] * port.closes[v][x])
        port.current_weights[k] = this_valuation / total_valuation

        # print k, v, port.normalized_weights[k], round(port.shares[k], 4), '\t', round(port.closes[v][x], 4), '\t', \
            # round(this_delta, 4), '\t', round(port.current_weights[k], 4)

    # print "Total Delta: ", total_delta
    logging.debug('%s %s %s' % (x, port.trimmed[port.assets[0]][x]['Date'], total_delta))

    return total_delta + port.starting_valuation

def get_drawdown(closes):

    all_time_high = closes[0]
    drawdown = [1.0]

    for k, close in enumerate(closes):
        if k == 0:
            continue
        all_time_high = max(close, all_time_high)
        drawdown.append(close / all_time_high)

    return drawdown

def get_data(port, base_etf, last_x_days = 0):
    """
    This section trims everything relative to SPY (so it will not have MORE data than SPY), but it can have less, so the
    lengths of the data are still not consistent yet.
    """
    print "start downloading"
    # The sort in data_retriever.py would corrupt the order of the asset list if it is not copied here
    asset_list = copy.deepcopy(port.assets)
    multithread_yahoo_download(thread_count=20, update_check=False, \
                               new_only=False, store_location = 'data/portfolio_analysis/', use_list=asset_list)
    load_redis(stock_list='tda_free_etfs.csv', db_number=1, file_location='data/portfolio_analysis/', dict_size=3, use_list=asset_list)
    stock_1_data = manage_redis.parse_fast_data(base_etf, db_to_use=1)
    logging.info('Loading Data...')
    logging.info('Base Start/End Dates: %s %s %s' % (base_etf, stock_1_data[0]['Date'],stock_1_data[-1]['Date']))
    for item in port.assets:
        logging.debug(item)
        stock_2_data = manage_redis.parse_fast_data(item, db_to_use=1)
        logging.info('Base Start/End Dates: %s %s %s' % (item, stock_2_data[0]['Date'], stock_2_data[-1]['Date']))
        stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
        port.closes[base_etf] = stock_1_close
        port.trimmed[base_etf] = stock_1_trimmed
        port.closes[item] = stock_2_close
        port.trimmed[item] = stock_2_trimmed

        #iteratively trim the input data
        stock_1_data = stock_1_trimmed

#        logging.info('%s %s %s %s %s %s' % (item, port.trimmed[item][0]['Date'], port.trimmed[item][-1]['Date'], \
#                                            base_etf, port.trimmed[base_etf][0]['Date'], port.trimmed[base_etf][-1]['Date']))

    # logging.info('%s \n %s \n %s \n %s' % (port.trimmed[item][0], port.trimmed[item][-1], port.trimmed[base_etf][0], port.trimmed[base_etf][-1]))

    # Now run all of the assets back through the trimming function but do it against the already-trimmed base etf
    stock_1_data = port.trimmed[base_etf]
    for item in port.assets:
        logging.debug(item)
        stock_2_data = port.trimmed[item]
        stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
        port.closes[base_etf] = stock_1_close
        port.trimmed[base_etf] = stock_1_trimmed
        port.closes[item] = stock_2_close
        port.trimmed[item] = stock_2_trimmed

        logging.info('%s %s %s %s %s %s' % (item, port.trimmed[item][0]['Date'], port.trimmed[item][-1]['Date'], \
                                    base_etf, port.trimmed[base_etf][0]['Date'], port.trimmed[base_etf][-1]['Date']))

        multiplier = 1
        if item in ['IEF', 'TLT']:
            multiplier = 3
        elif item in ['VNQ']:
            multiplier = 1
        elif item in ['GLD']:
            multiplier = 1

        if multiplier != 1:
            r = get_returns(stock_2_close)
            for k, v in enumerate(r):
                if k == 0:
                    continue
                new_val = ((1 + (multiplier*v)) * port.closes[item][k-1])

                port.closes[item][k] = round(new_val, 6)
                port.trimmed[item][k]['AdjClose'] = round(new_val, 6)

    ### Optional section that allows us to look at only the latest x days of data (faster to get a snapshot)
    if last_x_days != 0 and last_x_days < len(port.trimmed[port.assets[0]]):
        for item in port.assets:
            logging.debug(item)
            trimmed = port.trimmed[item][-(last_x_days+1):]
            closes = port.closes[item][-(last_x_days+1):]
#            trimmed = port.trimmed[item][-82:-17]
#            closes = port.closes[item][-82:-17]
            port.trimmed[item] = trimmed
            port.closes[item] = closes


    if not port.validate_portfolio():
        return False
    logging.info('\nData has been properly imported and validated.')
    return port

class MDPPortfolio():

    def __init__(self, assets_list):

        self.assets = assets_list
        self.weights = []
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
        # normalized_weights = np.array(self.normalized_weights)
        normalized_weights = weights
        transposed_weights = np.matrix.transpose(normalized_weights)
        weights = self.get_covariance_matrix(self.x)
        self.normalized_weights = weights

        # logging.info('weighted_vols_equal_one: transposed_weights: %s' % (transposed_weights))
        # logging.info('weighted_vols_equal_one: volatilities_matrix: %s' % (self.volatilities_matrix))

        r1 = np.dot(transposed_weights, self.volatilities_matrix)
        # logging.info('Weighted Vols Value: %s' % (r1))

        # print transposed_weights, self.volatilities_matrix
        # print weights, '\n', self.volatilities_matrix
        print "#### R1: \n", r1

        return np.array([r1[0] - 1])

    def get_covariance_matrix(self, x):
        end_index = x
        start_index = max(0, x - self.lookback)

#        if lookback == 0:
#            lookback = len(self.trimmed[self.assets[0]])
#        logging.info('Lookback Length: ' + str(lookback))

        for item in self.assets:
            closes = self.closes[item][start_index:end_index]
            self.past_returns[item] = get_returns(closes)

        for z in self.assets:
            rets = (self.past_returns[z])
            self.volatilities[z] = np.std(rets)

        # using (lookback-1) cuts out a leading zero that appears on day 1, since there is no return yet
        self.past_returns_matrix = np.array([self.past_returns[item] for item in self.assets])
    #    print "\nRETURNS: \n", self.returns_matrix

        volatilities = [ [self.volatilities[z]] for z in self.assets ]
        self.volatilities_matrix = np.array(volatilities)
        self.cov_matrix = np.cov(self.past_returns_matrix)

        self.inv_cov_matrix = scipy.linalg.inv(self.cov_matrix)
        self.transposed_volatilities_matrix = np.matrix.transpose(self.volatilities_matrix)

        # This basically re-creates equation 30.1 from the white paper
        numerator = np.dot(self.inv_cov_matrix, self.volatilities_matrix)
        denominator_a = np.dot(self.transposed_volatilities_matrix, self.inv_cov_matrix)
        denominator = np.dot(denominator_a, self.volatilities_matrix)
        self.weights = np.divide(numerator, denominator)

        # print numerator, '\n\n', denominator_a, '\n\n', denominator, '\n\n', self.weights

        # total_sum = sum(self.weights)[0]
        total_sum = sum([abs(n) for n in self.weights])[0]

        # print "\nSUM: ", total_sum, self.weights

        normalized_weights = np.divide(self.weights, total_sum)

        printed_weights = '\n'.join([ (str(a) + '\t' + str(normalized_weights[i][0])) for i, a in enumerate(self.assets)])
        ### print "\nNormalized Weights:\n", printed_weights
        ### print "\nSum of normalized: \n", sum(self.normalized_weights)[0]

        return normalized_weights

    def set_normalized_weights(self, weights, x):
        self.normalized_weights = weights
        self.current_entry_prices = [ self.closes[v][x] for k, v in enumerate(self.assets)]
        self.shares = [ (1000 * self.normalized_weights[k][0] / self.closes[v][x]) for k, v in enumerate(self.assets) ]

        

    def get_diversification_ratio(self):

        lookback = self.rebalance_time

        # transposed_weights = np.matrix.transpose(self.normalized_weights)
        transposed_weights = np.matrix.transpose(self.current_weights)
        numerator = np.dot(transposed_weights, self.volatilities_matrix)

        d1 = np.dot(transposed_weights, self.cov_matrix)
        d2 = np.dot(d1, self.normalized_weights)
        d3 = math.sqrt(abs(d2))

        ret = (numerator / d3)[0][0]

        self.diversification_ratio = ret

        return ret


def optimize(normalized_weights, cov_matrix):
    # print "NW, CM: ", normalized_weights, cov_matrix

    # from equation 31:
    # normalized_weights = np.array(port.normalized_weights)
    transposed_weights = np.matrix.transpose(normalized_weights)
    # _ = port.get_covariance_matrix(x)
    cov_matrix = cov_matrix
    r1 = np.dot(0.5, transposed_weights)
    r2 = np.dot(r1, cov_matrix)
    r3 = np.dot(r2, normalized_weights)

    # logging.info('Optimize function value: %s' % (str(r3)))

    ### Add in constraints here, make them positive and additive
    w = (abs((sum(normalized_weights) - 1)) + 1) ** 10    # This should get very large if the sum is <<1 or >>1

    neg_weights = abs(sum( [ a for a in normalized_weights if a < 0 ] ))

    neg_weights = (1 + neg_weights) ** 10
    if neg_weights != 1:
        # print "neg: ", neg_weights
        pass

    n = neg_weights if neg_weights else 0

    r3 = r3 + w + n

    return r3

class TradeLog(object):

    def __init__(self):
        self.Date = None
        self.Valuation = None
        self.DR = None
        self.Mean_Vol = None
        self.Weighted_Mean_Vol = None
        self.IsRebalanced = None



#    U.S. Stocks (SPY)
#    European Stocks (EFA)
#    Japanese Stocks (EWJ)
#    Emerging Market Stocks (EEM)
#    U.S. REITs (IYR)
#    International REITs (RWX)
#    U.S. Mid-term Treasuries (IEF)
#    U.S. Long-term Treasuries (TLT)
#    Commodities (DBC)
#    Gold (GLD)

