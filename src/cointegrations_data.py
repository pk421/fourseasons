import datetime
import copy
import numpy as np

from src import math_tools

import logging

logger = logging.getLogger(__name__)


def get_corrected_data(stock_1_data, stock_2_data, get_returns = None):
    """
    Performs the "trimming function" on stocks (see below). Returns "simple" returns after stocks are trimmed, Using
    the numpy array format. Also returns the full trimmed dictionary. Also calculates the ln returns of the simple
    returns (since it is convenient to do here).
    """
    stock_1_trimmed, stock_2_trimmed = trim_data(stock_1_data, stock_2_data)

    simple_1_close = np.empty(len(stock_1_trimmed))
    simple_2_close = np.empty(len(stock_2_trimmed))

    for k, v in enumerate(stock_1_trimmed):
        simple_1_close[k] = stock_1_trimmed[k]['AdjClose']
        simple_2_close[k] = stock_2_trimmed[k]['AdjClose']

    if get_returns:
        simple_1_returns = math_tools.get_ln_returns(simple_1_close)
        simple_2_returns = math_tools.get_ln_returns(simple_2_close)

    return simple_1_close, simple_2_close, stock_1_trimmed, stock_2_trimmed
    # return simple_1_returns, simple_2_returns

def trim_data(stock_1_data, stock_2_data):
    """
    This "trims" data on two stocks so that the two data sets will have the same start date and the same end date. It
    simply removes excess data from the stock with an earlier start date. It then checks that the length of the two
    sets of data. It tries to "propagate" the data to make the lengths match. If this fails, it raises an exception.
    Then, it not only checks lengths, but also checks that the actual dates used match each other. If one date is
    missing, it raises an exception.
    """

    stock_1_start = datetime.datetime.strptime(stock_1_data[0]['Date'], '%Y-%m-%d').date()
    stock_2_start = datetime.datetime.strptime(stock_2_data[0]['Date'], '%Y-%m-%d').date()
    stock_1_end = datetime.datetime.strptime(stock_1_data[-1]['Date'], '%Y-%m-%d').date()
    stock_2_end = datetime.datetime.strptime(stock_2_data[-1]['Date'], '%Y-%m-%d').date()

    if stock_1_start < stock_2_start:
        for x in xrange(0, len(stock_1_data)):
            # if datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date() >= \
            #    datetime.datetime.strptime(stock_1_data[0]['Date'], '%Y-%m-%d').date():
            if stock_1_data[x]['Date'] == stock_2_data[0]['Date']:
                trim_at = x
                break
        stock_1_data = stock_1_data[trim_at:]
        logger.debug('Stock 1 Start Trimmed: %s %s' % (stock_1_data[0]['Symbol'], trim_at))

    elif stock_2_start < stock_1_start:
        for x in xrange(0, len(stock_2_data)):
            # if datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date() >= \
            #    datetime.datetime.strptime(stock_1_data[0]['Date'], '%Y-%m-%d').date():
            if stock_2_data[x]['Date'] == stock_1_data[0]['Date']:
                trim_at = x
                break
        stock_2_data = stock_2_data[trim_at:]
        logger.debug('Stock 2 Start Trimmed: %s %s' % (stock_2_data[0]['Symbol'], trim_at))

    if stock_1_end < stock_2_end:
        for x in xrange(0, len(stock_2_data)):
            # if datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date() >= \
            #    datetime.datetime.strptime(stock_1_data[0]['Date'], '%Y-%m-%d').date():
            if stock_2_data[x]['Date'] == stock_2_data[-1]['Date']:
                trim_at = x
                break
        stock_2_data = stock_2_data[0:(trim_at+1)]
        logger.info('Stock 2 End Trimmed: %s %s' % (stock_2_data[0]['Symbol'], trim_at))

    elif stock_2_end < stock_1_end:
        for x in xrange(0, len(stock_1_data)):
            # if datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date() >= \
            #    datetime.datetime.strptime(stock_1_data[0]['Date'], '%Y-%m-%d').date():
            if stock_1_data[x]['Date'] == stock_2_data[-1]['Date']:
                trim_at = x
                break
        stock_1_data = stock_1_data[0:(trim_at+1)]
        logger.info('Stock 1 End Trimmed: %s %s' % (stock_1_data[0]['Symbol'], trim_at))

    if len(stock_1_data) != len(stock_2_data):
        stock_1_data, stock_2_data = propagate_on_fly(stock_1_data, stock_2_data)

    if len(stock_1_data) != len(stock_2_data) or \
        stock_2_data[len(stock_2_data)-1]['Date'] != stock_1_data[len(stock_1_data)-1]['Date'] or \
        stock_2_data[0]['Date'] != stock_1_data[0]['Date']:
        # print "\n**************"
        # print stock_1_data[0]['Symbol'], stock_2_data[0]['Symbol']
        # print len(stock_1_data), len(stock_2_data)
        # print stock_1_data[0]['Date'], stock_2_data[0]['Date']
        # print stock_1_data[len(stock_1_data)-1]['Date'], stock_2_data[len(stock_2_data)-1]['Date']
        # for x in xrange(len(stock_1_data)):
        # 		if stock_1_data[x]['Date'] != stock_2_data[x]['Date']:
        # 			print stock_1_data[x]['Date'], stock_2_data[x]['Date']
        # 			print "**************"
        # 			break
        e = stock_1_data[0]['Symbol'] + ' and ' + stock_2_data[0]['Symbol']
        raise Exception(e + ' did not trim properly and cannot be processed')

    # TODO: this section should be cythonized
    if len(stock_1_data) != len(stock_2_data):
        e = stock_1_data[0]['Symbol'] + ' and ' + stock_2_data[0]['Symbol']
        raise Exception(e + ' did not trim properly and cannot be processed')
    for x in xrange(len(stock_1_data)):
        if stock_1_data[x]['Date'] != stock_2_data[x]['Date']:
            e = stock_1_data[0]['Symbol'] + ' and ' + stock_2_data[0]['Symbol']
            print "\n\n"
            raise Exception(e + ' did not trim properly and cannot be processed')

    return stock_1_data, stock_2_data

def propagate_on_fly(stock_1_data, stock_2_data):
    """
    Searches through date data and compares dates between the two series. If it finds that one series skips a date that
    the other series had, it simply propagates the previous day's data to the missing date and uses the date the was
    found in the more complete data series.
    """
    x_min = min(len(stock_1_data), len(stock_2_data))
    x_max = max(len(stock_1_data), len(stock_2_data))

    for x in xrange(0, x_min):
        # if x > 3397 and x < 3407:
        # print x, stock_1_data[x]['Date'], stock_2_data[x]['Date'], stock_1_data[x]['Close'], stock_2_data[x]['Close']

        logger.debug('%s %s %s %s %s' % (x, x_min, x_max, len(stock_1_data), len(stock_2_data)))
        logger.debug('%s %s' % (stock_1_data[x]['Date'], stock_1_data[x]['Date']))

        if stock_1_data[x]['Date'] != stock_2_data[x]['Date']:
            if datetime.datetime.strptime(stock_1_data[x]['Date'], '%Y-%m-%d').date() > \
                datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date():
                temp = copy.deepcopy(stock_1_data[x-1])
                temp['Date'] = copy.deepcopy(stock_2_data[x]['Date'])
                stock_1_data.insert(x, temp)

            elif datetime.datetime.strptime(stock_2_data[x]['Date'], '%Y-%m-%d').date() > \
                datetime.datetime.strptime(stock_1_data[x]['Date'], '%Y-%m-%d').date():
                temp = copy.deepcopy(stock_2_data[x-1])
                temp['Date'] = copy.deepcopy(stock_1_data[x]['Date'])
                stock_2_data.insert(x, temp)

    # if we have reached the length of the shorter array, then we can just trim the longer one without propagation
    if len(stock_1_data) < len(stock_2_data):
        stock_2_data = stock_2_data[0:len(stock_1_data)]
    elif len(stock_2_data) < len(stock_1_data):
        stock_1_data = stock_1_data[0:len(stock_2_data)]

    return stock_1_data, stock_2_data



def get_paired_stock_list(stocks, fixed_stock=None):
    """
    Generates a list of dictionaries. Each dictionary contains 2 stocks - a pair. Using the fixed stock option will
    basically just create a pairing of all stocks with that one stock. Otherwise, it creates every pair possible.
    """

    len_stocks = len(stocks)
    paired_list = []
    if fixed_stock is not None:
        for x in xrange(0, len_stocks):
            item = {'stock_1' : fixed_stock, 'stock_2' : stocks[x], 'corr_coeff' : 0.0, 'beta' : 0.0}
            paired_list.append(item)
        print "Paired list length: ", len(paired_list)
        return paired_list

    for x in xrange(0, len_stocks):
        stock_1 = stocks[x]

        for y in xrange(x+1, len_stocks):
            stock_2 = stocks[y]

            item = {'stock_1' : stock_1, 'stock_2' : stock_2, 'corr_coeff' : 0.0, 'beta' : 0.0}
            paired_list.append(item)

    print "Paired list length: ", len(paired_list)
    return paired_list


def get_bunches_of_pairs():

    """
    Data Failures:
    BA, BRIS, BRIL, HPQ, IDZ, KO, LNKD, MNST, RUT
    """

    tech = ('MSFT', 'AAPL', 'ORCL', 'NVDA', 'GOOG', 'ADBE', 'CSCO', 'CRM', 'IBM', 'SMDK', 'SYMC', 'MSI')
    banks = ('JPM', 'GS', 'BAC', 'C', 'WFC', 'BBT', 'MS')
    oil_majors = ('XOM', 'COP', 'CVX', 'RIG', 'HAL', 'OIL', 'SLB', 'APA', 'APC', 'CHK', 'DVN', 'BHI')
    gold = ('GLD', 'SLV', 'ABX', 'NEM', 'SLW', 'AUY', 'KGC', 'GOLD', 'GDX', 'GDXJ', 'AEM', 'GG')
    developed = ('SPY', 'DIA', 'SDY', 'QQQ', 'EWA', 'EWP', 'EEM', 'VGK', 'EWG', 'FEZ', 'EWL', 'EWI', 'EWQ', 'EWD', 'EWN')
    developing = ('EEM', 'DEM', 'FXI', 'MCHI', 'GXG', 'EWZ')
    defense = ('LMT', 'RTN', 'NOC', 'TXT', 'GD')
    drinks = ('CCE', 'DPS', 'PEP', 'ABV', 'BUD', 'TAP')
    health_insurance = ('A', 'CI')
    drugs = ('AZN', 'BMY', 'GSK', 'JNJ', 'LLY', 'MRK', 'NVS', 'PFE')
    utilities = ('D', 'DUK', 'ED', 'PCG', 'ETR')


    leverage = ('SPY', 'SSO', 'SDS', 'TBT', 'QLD', 'QID', 'UYG', 'DIG', 'DDM', 'UYM', 'DUST', 'NUGT', \
                'INDL', 'GDX', 'GDXJ', 'EZJ', 'EFO')

    lists_to_use = (tech, banks, oil_majors, gold, developed, defense, drinks, health_insurance, drugs, leverage)
    # lists_to_use = (health_insurance, defense)

    paired_list = []
    for item in lists_to_use:

        pairs = get_paired_stock_list(item)
        paired_list.extend(pairs)

    print len(paired_list)
    return paired_list


