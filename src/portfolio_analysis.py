import numpy as np
import scipy as scipy
import datetime
import time

import math

import toolsx as tools
from data.redis import manage_redis
from src.cointegrations_data import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly, \
                                    get_bunches_of_pairs

from math_tools import get_returns

import logging

def run_portfolio_analysis():

    portfolio = Portfolio()

    assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
#    assets_list = ['SPY', 'GLD', 'EEM', 'TLT']
#    assets_list = ['SPY', 'SHY', 'GLD']
    assets_list = ['TNA', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'SLV']

    assets_list = ['SPY', 'SSO', 'TNA', 'IEF']

    logging.root.setLevel(logging.INFO)
    logging.debug(str(assets_list))

    # we implicitly assume that SPY has correct data and use it to baseline the other instruments
    stock_1_data = manage_redis.parse_fast_data('SPY', db_to_use=0)
    input_data = get_data(assets_list, stock_1_data)

    covariance_matrix = get_covariance_matrix(input_data, assets_list, lookback=4)

    



    return

def get_covariance_matrix(input_data, assets_list, lookback=0):
    if lookback == 0:
        lookback = len(input_data[assets_list[0]]['trimmed'])
    logging.info('Lookback Length: ' + str(lookback))

    for item in assets_list:
        closes = input_data[item]['closes']
        input_data[item]['returns'] = get_returns(closes[-lookback:])
        #print "***", item, closes, input_data[item]['returns']


    returns_items = []
    volatilities = []
    for z in assets_list:
        #print z, input_data[item]['returns']
        rets = (input_data[z]['returns'])
        returns_items.append(rets)
        volatility = [np.std(rets)]
        volatilities.append(volatility)
    #print returns_items

    # using (lookback-1) cuts out a leading zero that appears on day 1, since there is no return yet
    returns_matrix = np.array([input_data[item]['returns'][-(lookback-1):] for item in assets_list])
    print "\nRETURNS: \n", returns_matrix
    

    # print '\n', np.cov(input_data['SPY']['returns'])

    volatilities = np.array(volatilities)
    cov_matrix = np.cov(returns_items)
    inv_cov_matrix = scipy.linalg.inv(cov_matrix)
    transposed_volatilities = np.matrix.transpose(volatilities)


    print "\nCOV MATRIX: \n", cov_matrix
    print "\nINVERSE COV MATRIX: \n", inv_cov_matrix
    print "\nVolatilties: \n", volatilities
    print "\nTransposed Volatilities: \n", transposed_volatilities
    print "\n"

    numerator = np.dot(inv_cov_matrix, volatilities)
    denominator_a = np.dot(transposed_volatilities, inv_cov_matrix)
    denominator = np.dot(denominator_a, volatilities)
    weights = np.divide(numerator, denominator)

    print numerator, '\n\n', denominator_a, '\n\n', denominator, '\n\n', weights

    total_sum = sum(weights)[0]

    print "\nSUM: ", total_sum

    normalized_weights = np.divide(weights, total_sum)

    print "\nNormalized Weights:\n", normalized_weights


    port_constraints = [{'type': 'eq', 'fun': positive_sum_only},\
                        {'type': 'eq', 'fun': result_positive}]

    # get_mean_variance(matrix)
    # result = scipy.optimize.minimize(get_mean_variance, [0.5,0.5], method='TNC', options={'xtol': 1e-8, 'disp': True}, bounds = ((0, None), (0,None)))
#    print result
#    print result.x


def get_mean_variance(matrix):

    logging.debug('MATRIX: %s' % (matrix))

    covariances = [1.2, -0.6]
    dot_product = matrix * covariances
    total = sum(dot_product)

    total = 0 + abs(total)

#    total = 0
#    for x in xrange(0, 2):
#        total += matrix[x]

#    total = total / len(covariances)

    logging.debug('Total: %s' % (total))
    return total

def positive_sum_only(matrix):
    k = 1.0
    result = k - sum(matrix)

    logging.info('Constraint: %s' % (result))

    return result

def result_positive(matrix):
    k = 0.0

    return min(0, (k-matrix[0]))

    



def get_data(assets_list, stock_1_data, base_etf='SPY'):
    """
    Before this point everything was trimmed relative to SPY. Now, we trim everything so all datasets are same length.
    """
    input_data = {}

    logging.info('Loading Data...')
    for item in assets_list:
        logging.debug(item)
        stock_2_data = manage_redis.parse_fast_data(item, db_to_use=0)
        stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
        input_data['SPY'] = {}
        input_data[item] = {}
        input_data['SPY']['closes'] = stock_1_close
        input_data['SPY']['trimmed'] = stock_1_trimmed
        input_data[item]['closes'] = stock_2_close
        input_data[item]['trimmed'] = stock_2_trimmed

    logging.debug('%s %s' % (input_data[item]['trimmed'][0], input_data['SPY']['trimmed'][0]))


    max_start_date = 0
    etf_latest_start = 'SPY'

    for item in assets_list:
#        print item, int(input_data[item]['trimmed'][0]['Date'].replace('-', ''))
        start_date = int(input_data[item]['trimmed'][0]['Date'].replace('-', ''))
        if start_date > max_start_date and item != 'SPY':
            max_start_date = start_date
            etf_latest_start = item

    # logging.debug('%s %s' % (input_data[etf_latest_start]['trimmed'][0], input_data['SPY']['trimmed'][0]))

    # run SPY thru trimming algo to reset its start date, then re-run all others against SPY to reset them
    etf_close, base_close, etf_trim, base_trim = get_corrected_data(input_data[etf_latest_start]['trimmed'], input_data['SPY']['trimmed'])

    input_data['SPY']['closes'] = base_close
    input_data['SPY']['trimmed'] = base_trim

    stock_1_data = input_data['SPY']['trimmed']
    for item in assets_list:
        logging.info(item)
        stock_2_data = input_data[item]['trimmed']
        stock_1_close, stock_2_close, stock_1_trimmed, stock_2_trimmed = get_corrected_data(stock_1_data, stock_2_data)
        input_data['SPY'] = {}
        input_data[item] = {}
        input_data['SPY']['closes'] = stock_1_close
        input_data['SPY']['trimmed'] = stock_1_trimmed
        input_data[item]['closes'] = stock_2_close
        input_data[item]['trimmed'] = stock_2_trimmed

    for item in assets_list:
        logging.info('\n%s %s' % (item, int(input_data[item]['trimmed'][0]['Date'].replace('-', ''))))
        logging.info('%s %s' % (item, input_data[item]['trimmed'][25]))

    return input_data



class Portfolio():

    def __init__(self):
        pass





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