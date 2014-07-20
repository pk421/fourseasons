import numpy as np
import scipy as scipy
import datetime
import time

import math

import toolsx as tools
from data.redis import manage_redis
from src.cointegrations_data import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly, \
                                    get_bunches_of_pairs

from src.indicator_system       import get_sharpe_ratio

from math_tools import get_returns

import logging
logging.root.setLevel(logging.INFO)

def run_portfolio_analysis():

    assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
    assets_list = ['SPY', 'TLT', 'GLD']
    # assets_list = ['TNA', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'SLV']
   # assets_list = ['SPY', 'GLD']

    in_file_name = 'list_dow_modified'
    location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + in_file_name + '.csv'
    in_file = open(location, 'r')
    stock_list = in_file.read().split('\n')
    for k, item in enumerate(stock_list):
        new_val = item.split('\r')[0]
        stock_list[k] = new_val
    in_file.close()

#    assets_list = stock_list



#    matrix = np.array([[0.5],[0.5]])
#    port_constraints = [{'type': 'eq', 'fun': no_leverage},\
#                        {'type': 'eq', 'fun': result_positive}]
#
#    get_mean_variance(matrix)
#    result = scipy.optimize.minimize(get_mean_variance, [0.5,0.5], method='SLSQP', options={'xtol': 1e-8, 'disp': True}, bounds = ((0, 1), (0,1)), constraints=port_constraints)
#    print result
#    print result.x
#    return

    mdp_port = MDPPortfolio(assets_list=assets_list)

    logging.debug(str(mdp_port.assets))
#    stock_1_data = manage_redis.parse_fast_data(mdp_port.assets[0], db_to_use=0)
    mdp_port = get_data(mdp_port, base_etf=mdp_port.assets[0])

    if not mdp_port:
        logging.warning("Get Data returned False, Failure")
        return False

    rebalance_time = 30
    # mdp_port.weights = [ [1.0 / len(mdp_port.assets)] for x in mdp_port.assets]
    mdp_port.weights = [ [.99], [0.005], [0.005]]
    mdp_port.weights = np.array(mdp_port.weights)
    mdp_port.normalized_weights = mdp_port.weights


    current_portfolio_valuation = get_port_valuation(mdp_port, x=0)

    x=0
    previous_rebalance = 1
    while x < len(mdp_port.trimmed[mdp_port.assets[0]]):
#        if x == 0:
#            x += 1
#            continue
        if not (x % rebalance_time == 0) or x == 0:
            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)
            mdp_port.portfolio_valuations.append((mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance))
            print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance
            x += 1
            continue
        else:
            # When calculating the rebalance ratio, we first assume we used the old weights for today's valuation. Then
            # we calculate new weights for today, value the portfolio for today, then find the ratio for today if we
            # had used the old weights for today

            old_weighted_valuation = get_port_valuation(mdp_port, x=x) / previous_rebalance
            print "old weighted: ", old_weighted_valuation

            lookback_val = min(x, 126)
            ### mdp_port.rebalance(x, lookback=lookback_val)

            mdp_port.x = x
            # mdp_port.optimize(x, lookback=lookback_val)
            port_constraints = [{'type': 'eq',
                                 'fun': mdp_port.weighted_vols_equal_one,
                                 # 'jac':
                               }]

#            result = scipy.optimize.minimize(mdp_port.optimize, mdp_port.weights, method='SLSQP', options={'xtol': 1e-8, 'disp': True}, bounds = [(0,1) for z in mdp_port.assets] , constraints=port_constraints)
            result = scipy.optimize.minimize(mdp_port.optimize, mdp_port.weights, jac=mdp_port.optimize_derivative, method='SLSQP', options={'xtol': 1e-8, 'disp': True}, constraints=port_constraints)
            print "**************FINAL RESULT: \n", result
            return

            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)

            previous_rebalance = current_portfolio_valuation / old_weighted_valuation
            print "Previous Rebalance: ", previous_rebalance

            mdp_port.portfolio_valuations.append((mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance))
            print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation / previous_rebalance
            x+=1

    print "len: ", len(mdp_port.trimmed[mdp_port.assets[0]]), len(mdp_port.portfolio_valuations)

    print mdp_port.trimmed[mdp_port.assets[0]][0], mdp_port.trimmed[mdp_port.assets[0]][-1]
    print '\n', mdp_port.portfolio_valuations[0], mdp_port.portfolio_valuations[-1]

    sharpe_price_list = []
    for val in mdp_port.portfolio_valuations:
        sharpe_price_list.append(('existing_trade', 'long', val[1]))

    smean, sstd, sneg_std, spos_std, ssharpe, ssortino, savg_loser, savg_winner, spct_losers = get_sharpe_ratio(sharpe_price_list)

    print '\t\tSystem:'
    print 'ArithMu: \t', round(smean, 6)
    print 'Sigma: \t\t', round(sstd, 6)
    print 'NegSigma: \t', round(sneg_std, 6)
    print 'NegSigma/Tot: \t', round((sneg_std/sstd), 6)
    print 'Sharpe: \t', round(ssharpe, 6)
    print 'Sortino: \t', round(ssortino, 6)



#    normalized_weights = mdp_port.get_covariance_matrix(x=len(mdp_port.trimmed[mdp_port.assets[0]]), lookback=50)

    return

def get_port_valuation(port, x=0):

    ### TODO: accept a parameter for rebalance ratio so that it is not divided everywhere else

    weights = port.normalized_weights
    total_valuation = 0

    for k, v in enumerate(port.assets):
        total_valuation += weights[k][0] * port.closes[v][x]
        # print k, v, weights[k], port.closes[v][x], weights[k][0] * port.closes[v][x]

    logging.debug('%s %s %s' % (x, port.trimmed[port.assets[0]][x]['Date'], total_valuation))

    return total_valuation






def get_mean_variance(matrix):

    logging.debug('MATRIX: %s' % (matrix))

    covariances = [1.2, 0.6]
    dot_product = matrix * covariances
    total = sum(dot_product)

    total = 0 + abs(total)

#    total = 0
#    for x in xrange(0, 2):
#        total += matrix[x]

#    total = total / len(covariances)

    logging.debug('Total: %s' % (total))
    return total

def no_leverage(matrix):
    k = 1.0
    result = k - sum(matrix)

    logging.info('Constraint: %s' % (result))

    return result

def result_positive(matrix):
    k = 0.0

    return min(0, (k-matrix[0]))

#    port_constraints = [{'type': 'eq', 'fun': positive_sum_only},\
#                        {'type': 'eq', 'fun': result_positive}]

    # get_mean_variance(matrix)
    # result = scipy.optimize.minimize(get_mean_variance, [0.5,0.5], method='TNC', options={'xtol': 1e-8, 'disp': True}, bounds = ((0, None), (0,None)))
#    print result
#    print result.x



def get_data(port, base_etf):
    """
    This section trims everything relative to SPY (so it will not have MORE data than SPY), but it can have less, so the
    lengths of the data are still not consistent yet.
    """
    stock_1_data = manage_redis.parse_fast_data(base_etf, db_to_use=0)

    logging.info('Loading Data...')
    logging.info('Base Start/End Dates: %s %s %s' % (base_etf, stock_1_data[0]['Date'],stock_1_data[-1]['Date']))
    for item in port.assets:
        logging.debug(item)
        stock_2_data = manage_redis.parse_fast_data(item, db_to_use=0)
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

        self.universal_lookback = 90
        self.rebalance_time = 30

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

    def rebalance(self, x, lookback=30):

        current_weights = self.normalized_weights
        current_valuation = get_port_valuation(self, x)
        new_weights = self.get_covariance_matrix(x, lookback=lookback)
        return True

#    def optimize(self, x, lookback=30):
    def optimize(self, matrix):

        lookback = 30
        x = self.x

        # from equation 31
        normalized_weights = np.array(self.normalized_weights)
        transposed_weights = np.matrix.transpose(normalized_weights)
        _ = self.get_covariance_matrix(x, lookback=lookback)
        cov_matrix = self.cov_matrix
        # ret = 0.5 * transposed_weights * cov_matrix * self.normalized_weights
        r1 = np.dot(0.5, transposed_weights)
        r2 = np.dot(r1, cov_matrix)
        r3 = np.dot(r2, normalized_weights)

        # print "HERE: "
        # print normalized_weights
        # print transposed_weights
        # print cov_matrix

        # print r1
        # print r2
        print "OPTIMIZE RESULT: \n", r3

        return r3[0][0]

    def optimize_derivative(self, weights):
        print "\nOptimize Derivative:"
        print "Weights: ", weights

        _ = self.get_covariance_matrix(self.x, lookback=30)
        cov_matrix = self.cov_matrix
        print self.cov_matrix

        derivative = np.array([1, 2, 3])

        len_data = len(weights)
        deriv_items = []
        for row in xrange(0, len_data):
            items = [ 2 * weights[x] * cov_matrix[row][x] for x in xrange(0, len_data) ]
            print "*******ROW: ", row, items
            items_sum = sum(items)
            deriv_items.append(items_sum)

        derivative = np.array(deriv_items)
        print "$$$$$$$$$ FINAL DERIV ITEMS: ", derivative




        return derivative

    def weighted_vols_equal_one(self, weights):
        # normalized_weights = np.array(self.normalized_weights)
        normalized_weights = weights
        transposed_weights = np.matrix.transpose(normalized_weights)
        _ = self.get_covariance_matrix(self.x, lookback=30)

        r1 = np.dot(transposed_weights, self.volatilities_matrix)

        # print transposed_weights, self.volatilities_matrix
        print weights, '\n', self.volatilities_matrix
        print "#### R1: \n", r1


        return r1[0] - 1

    def get_covariance_matrix(self, x, lookback=0):

        end_index = x
        start_index = x - lookback

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


    #    print "\nCOV MATRIX: \n", self.cov_matrix
    #    print "\nINVERSE COV MATRIX: \n", self.inv_cov_matrix
    #    print "\nVolatilties: \n", self.volatilities_matrix
    #    print "\nTransposed Volatilities: \n", self.transposed_volatilities_matrix
    #    print "\n"

        # This basically re-creates equation 30.1 from the white paper
        numerator = np.dot(self.inv_cov_matrix, self.volatilities_matrix)
        denominator_a = np.dot(self.transposed_volatilities_matrix, self.inv_cov_matrix)
        denominator = np.dot(denominator_a, self.volatilities_matrix)
        self.weights = np.divide(numerator, denominator)

        # print numerator, '\n\n', denominator_a, '\n\n', denominator, '\n\n', self.weights

        total_sum = sum(self.weights)[0]

        # print "\nSUM: ", total_sum

        self.normalized_weights = np.divide(self.weights, total_sum)


        printed_weights = '\n'.join([ (str(a) + '\t' + str(self.normalized_weights[i][0])) for i, a in enumerate(self.assets)])
        print "\nNormalized Weights:\n", printed_weights
        print "\nSum of normalized: \n", sum(self.normalized_weights)[0]

        return self.normalized_weights


    #    port_constraints = [{'type': 'eq', 'fun': positive_sum_only},\
    #                        {'type': 'eq', 'fun': result_positive}]

        # get_mean_variance(matrix)
        # result = scipy.optimize.minimize(get_mean_variance, [0.5,0.5], method='TNC', options={'xtol': 1e-8, 'disp': True}, bounds = ((0, None), (0,None)))
    #    print result
    #    print result.x


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