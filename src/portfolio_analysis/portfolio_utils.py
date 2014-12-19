import copy

import logging
logging.root.setLevel(logging.INFO)

from data.redis                 import manage_redis
from src.data_retriever         import load_redis
from src.data_retriever         import multithread_yahoo_download
from src.cointegrations_data    import get_paired_stock_list, get_corrected_data, trim_data, propagate_on_fly, \
                                    get_bunches_of_pairs
from src.math_tools             import get_returns, get_ln_returns


def get_data(port, base_etf, last_x_days = 0, get_new_data=True):
    """
    This section trims everything relative to SPY (so it will not have MORE data than SPY), but it can have less, so the
    lengths of the data are still not consistent yet.
    """
    print "start downloading"
    # The sort in data_retriever.py would corrupt the order of the asset list if it is not copied here
    asset_list = copy.deepcopy(port.assets)
    if get_new_data:
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
        elif item in ['IWM', 'DIA', 'SPY', 'EFA', 'VWO']:
            multiplier = 3
        elif item in ['VNQ', 'IYR']:
            multiplier = 3
        elif item in ['GLD', 'SLV', 'OIL']:
            multiplier = 3
        elif item in ['DBC']:
            multiplier = 2

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
#            trimmed = port.trimmed[item][-64:-1]
#            closes = port.closes[item][-64:-1]
            port.trimmed[item] = trimmed
            port.closes[item] = closes


    if not port.validate_portfolio():
        return False
    logging.info('\nData has been properly imported and validated.')
    return port