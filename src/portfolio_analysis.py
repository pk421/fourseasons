import numpy as np
import scipy as scipy
import datetime
import time

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

def run_portfolio_analysis():

    # Brute Force Calcs = # nodes ^ # assets
    assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
    # assets_list = ['TNA', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'SLV']
#    assets_list = ['SPXL', 'TYD', 'DRN', 'DGP', 'EDC']
#    assets_list = ['SPY', 'IEF', 'VNQ', 'GLD', 'EEM']

    # 401k
    # Note: VBMPX has a shorter duration and is actually less volatile than VIPIX
    # VEMPX is small caps basically but has not done as well in this portfolio compared to VIIIX
    # The five asset list here is less volatile and better diversified, but has a lower return
    # assets_list = ['VIIIX', 'VEMPX', 'VTPSX', 'VIPIX', 'VBMPX']
    # assets_list = ['VIIIX', 'VTPSX', 'VIPIX']

    in_file_name = 'list_dow_modified'
    location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + in_file_name + '.csv'
    in_file = open(location, 'r')
    stock_list = in_file.read().split('\n')
    for k, item in enumerate(stock_list):
        new_val = item.split('\r')[0]
        stock_list[k] = new_val
    in_file.close()

    mdp_port = MDPPortfolio(assets_list=assets_list)

    logging.debug(str(mdp_port.assets))
    mdp_port = get_data(mdp_port, base_etf=mdp_port.assets[0], last_x_days=0)

    if not mdp_port:
        logging.warning("Get Data returned False, Failure")
        return False

    mdp_port.weights = [ [1.0 / len(mdp_port.assets)] for x in mdp_port.assets]
    mdp_port.weights = np.array(mdp_port.weights)
    mdp_port.normalized_weights = mdp_port.weights

    x=0
    mdp_port.x = x
    mdp_port.lookback = 63
    mdp_port.rebalance_time = 63
    mdp_port.rebalance_counter = 0
    previous_rebalance = 1
    while x < len(mdp_port.trimmed[mdp_port.assets[0]]):
        if mdp_port.rebalance_counter < mdp_port.rebalance_time:

            mdp_port.x = x
            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)
            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance])
            print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance
            mdp_port.rebalance_counter += 1

            if x > mdp_port.rebalance_time:
                # The statement below DOES have an impact on whether a rebalance is hit in this if block
                # It also affects overall program flow and the end result
                _ = mdp_port.get_covariance_matrix(x)
                trailing_diversification_ratio = mdp_port.get_diversification_ratio()
                print "Trailing DR: ", x, trailing_diversification_ratio

                # Disable this if block to revert to previous behavior
                if trailing_diversification_ratio < 1.5 and mdp_port.rebalance_counter > (mdp_port.rebalance_time / 2):
                    # Force the update to occur if the div ratio is low and we've waited a minimum time since last rebal
                    mdp_port.rebalance_counter = mdp_port.rebalance_time
            x += 1
            continue
        else:
            mdp_port.rebalance_counter = 0
            # When calculating the rebalance ratio, we first assume we used the old weights for today's valuation. Then
            # we calculate new weights for today, value the portfolio for today, then find the ratio for today if we
            # had used the old weights for today
            old_weighted_valuation = get_port_valuation(mdp_port, x=x) / previous_rebalance
            # This is the portfolio value today, IF we had been using the previous weighting
            print "Today's value @ old weighting: ", old_weighted_valuation
            mdp_port.x = x

            ###
            mdp_port, normalized_weights = do_optimization(mdp_port, x, method='basinhopping')
            # mdp_port, normalized_weights = do_optimization(mdp_port, x, method='brute')
            ###

            trailing_diversification_ratio = mdp_port.get_diversification_ratio()
            mdp_port.trailing_DRs.append(trailing_diversification_ratio)
            print "Trailing Diversification Ratio: ", x, trailing_diversification_ratio

            print "Constrained (+) Result: \n", normalized_weights
            print "weighted vols equal one check: ", mdp_port.weighted_vols_equal_one(normalized_weights)
            mdp_port.set_normalized_weights(normalized_weights)

            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)
            print "Current Portfolio New Valuation: ", current_portfolio_valuation
            previous_rebalance = current_portfolio_valuation / old_weighted_valuation
            print "Previous Rebalance: ", previous_rebalance

            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance])
            print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance
            x+=1


    for i, p in enumerate(mdp_port.portfolio_valuations):
        if i == 0:
            starting_value = p[1]
        mdp_port.portfolio_valuations[i][1] = mdp_port.portfolio_valuations[i][1] / starting_value


    if len(mdp_port.trimmed[mdp_port.assets[0]]) != len(mdp_port.portfolio_valuations):
        raise Exception('Length of asset list data does not match length of valuations data.')

    # print mdp_port.trimmed[mdp_port.assets[0]][0], mdp_port.trimmed[mdp_port.assets[0]][-1]
    print '\n', mdp_port.portfolio_valuations[0], mdp_port.portfolio_valuations[-1]

    sharpe_price_list = []
    for val in mdp_port.portfolio_valuations:
        sharpe_price_list.append(('existing_trade', 'long', val[1]))

    smean, sstd, sneg_std, spos_std, ssharpe, ssortino, savg_loser, savg_winner, spct_losers = get_sharpe_ratio(sharpe_price_list)


    output_fields = ('Date', 'Valuation', 'DR', 'Mean_Vol', 'Weighted_Mean_Vol', 'IsRebalanced')

    output_string = '\n'.join( [(str(n[0]) + ',' + str(n[1])) for n in mdp_port.portfolio_valuations] )
    current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_file_name = '/home/wilmott/Desktop/fourseasons/fourseasons/results/portfolio_analysis' + '_' + str(current_time) +'.csv'
    with open(out_file_name, 'w') as f:
        f.write(output_string)


    print "Mean Diversification Ratio: ", np.mean(mdp_port.trailing_DRs), len(mdp_port.trailing_DRs)
    print '\t\tSystem:'
    print 'ArithMu: \t', round(smean, 6)
    print 'Sigma: \t\t', round(sstd, 6)
    print 'NegSigma: \t', round(sneg_std, 6)
    print 'NegSigma/Tot: \t', round((sneg_std/sstd), 6)
    print 'Sharpe: \t', round(ssharpe, 6)
    print 'Sortino: \t', round(ssortino, 6)


    ref_log = []
    spy_price_list = mdp_port.closes[mdp_port.assets[0]]
    for val in spy_price_list:
        ref_log.append(('existing_trade', 'long', val))

    smean, sstd, sneg_std, spos_std, ssharpe, ssortino, savg_loser, savg_winner, spct_losers = get_sharpe_ratio(ref_log)

    print '\t\tReference:'
    print 'ArithMu: \t', round(smean, 6)
    print 'Sigma: \t\t', round(sstd, 6)
    print 'NegSigma: \t', round(sneg_std, 6)
    print 'NegSigma/Tot: \t', round((sneg_std/sstd), 6)
    print 'Sharpe: \t', round(ssharpe, 6)
    print 'Sortino: \t', round(ssortino, 6)


    print "\nFinished: "
    print "File Written: ", out_file_name.split('/')[-1]

    return

def do_optimization(mdp_port, x, method='basinhopping'):

#   result = scipy.optimize.minimize(mdp_port.optimize, mdp_port.weights, jac=mdp_port.optimize_jacobian, method='SLSQP', options={'xtol': 1e-8, 'disp': True}, bounds = [(0,1) for z in mdp_port.assets] , constraints=port_constraints)

    nw = mdp_port.normalized_weights
    # This updates the covariance matrix using the current volatilities, so we can optimize based on that
    theoretical_weights = mdp_port.get_covariance_matrix(x)
    cm = mdp_port.cov_matrix
    assets = mdp_port.assets

    if method == 'brute':
        result = scipy.optimize.brute(optimize, [(0,1) for z in assets], (cm,), Ns=21, full_output=False)
        sum_result = sum(result)
        normalized_weights = np.array([[round(z / sum_result, 4)] for z in result])
        print result[0], sum_result


    if method == 'basinhopping':
        mk = {'args':(cm,)}
        
        result = scipy.optimize.basinhopping(optimize, nw, niter=20, disp=True, minimizer_kwargs=mk)
        # result = scipy.optimize.basinhopping(optimize, nw, niter=50, T=5e-2, stepsize=5e-2, disp=True, minimizer_kwargs=mk)

        sum_result = sum(result.x)
        normalized_weights = np.array([[round(z / sum_result, 4)] for z in result.x])

    print result

    print "Theoretical Result: \n", theoretical_weights
    print "weighted vols equal one check: ", mdp_port.weighted_vols_equal_one(mdp_port.normalized_weights)

    ###
    # Letting this code executes overrides the optimization and uses the theoretical values for port analysis
    # normalized_weights = mdp_port.normalized_weights
    ###

    return mdp_port, normalized_weights


def get_port_valuation(port, x=0):
    ### TODO: accept a parameter for rebalance ratio so that it is not divided everywhere else

    weights = port.normalized_weights
    total_valuation = 0

    for k, v in enumerate(port.assets):
        total_valuation += weights[k][0] * port.closes[v][x]
        # print k, v, weights[k], port.closes[v][x], weights[k][0] * port.closes[v][x]
    logging.debug('%s %s %s' % (x, port.trimmed[port.assets[0]][x]['Date'], total_valuation))

    return total_valuation

def update_trade_log(port, x):



    return trade_log

def get_data(port, base_etf, last_x_days = 0):
    """
    This section trims everything relative to SPY (so it will not have MORE data than SPY), but it can have less, so the
    lengths of the data are still not consistent yet.
    """
    print "start downloading"
    multithread_yahoo_download(thread_count=20, update_check=False, \
                               new_only=False, store_location = 'data/portfolio_analysis/', use_list=port.assets)

    load_redis(stock_list='tda_free_etfs.csv', db_number=1, file_location='data/portfolio_analysis/', dict_size=3, use_list=port.assets)

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


    ### Optional section that allows us to look at only the latest x days of data (faster to get a snapshot)
    if last_x_days != 0 and last_x_days < len(port.trimmed[port.assets[0]]):
        for item in port.assets:
            logging.debug(item)
            trimmed = port.trimmed[item][-last_x_days:]
            closes = port.closes[item][-last_x_days:]
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

#    def optimize_jacobian(self, weights):
#        # logging.info('Optimization Derivative Weights: %s' % (weights))
#        _ = self.get_covariance_matrix(self.x)
#        cov_matrix = self.cov_matrix
#
#        len_data = len(weights)
#        deriv_items = []
#        for row in xrange(0, len_data):
#            items = [ 2 * weights[x] * cov_matrix[row][x] for x in xrange(0, len_data) ]
#            # logging.info('Derivative Row Items %s %s' % (row, items))
#            items_sum = sum(items)
#            deriv_items.append(items_sum)
#
#        derivative = np.array(deriv_items)
#        # logging.info('Final Derivative %s' % (derivative))
#
#        return derivative

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

#    def weighted_vols_equal_one_jacobian(self, weights):
#        len_weights = len(weights)
#        ret = len_weights * [1]
#        # logging.info('Weighted vols equal one jac: %s' % (ret))
#        ret = np.array(ret)
#        return ret

#    def no_negative_weights(self, weights):
#
#        negative_sum = sum([ x for x in weights if x < 0 ])
#        if negative_sum > 0:
#            raise Exception('Negative Sum is greater than zero: %s' % (str(negative_sum)))
#        logging.info("Negative Sum: %s" % (str(negative_sum)))
#
#        return negative_sum
#
#    def no_negative_weights_jacobian(self, weights):
#        len_weights = len(weights)
#        ret = len_weights * [1]
#        logging.info('No negative weights jac: %s' % (ret))
#        return ret

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

        total_sum = sum(self.weights)[0]

        # print "\nSUM: ", total_sum

        normalized_weights = np.divide(self.weights, total_sum)

        printed_weights = '\n'.join([ (str(a) + '\t' + str(normalized_weights[i][0])) for i, a in enumerate(self.assets)])
        ### print "\nNormalized Weights:\n", printed_weights
        ### print "\nSum of normalized: \n", sum(self.normalized_weights)[0]

        return normalized_weights

    def set_normalized_weights(self, weights):
        self.normalized_weights = weights

    def get_diversification_ratio(self):

        lookback = self.rebalance_time

        transposed_weights = np.matrix.transpose(self.normalized_weights)
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
    w = (abs((sum(normalized_weights) - 1)) + 1) ** 6    # This should get very large if the sum is <<1 or >>1

    neg_weights = abs(sum( [ a for a in normalized_weights if a < 0 ] ))

    neg_weights = (1 + neg_weights) ** 6
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

