__author__ = 'Mike'

import numpy as np

import logging

def get_drawdown(closes):

    all_time_high = closes[0]
    drawdown = [1.0]

    for k, close in enumerate(closes):
        if k == 0:
            continue
        all_time_high = max(close, all_time_high)
        drawdown.append(close / all_time_high)

    return drawdown

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

    port.current_weights = np.array([ w for w in port.current_weights ])

    return net_long_valuation

class TradeLog(object):

    def __init__(self):
        self.Date = None
        self.Valuation = None
        self.DR = None
        self.Mean_Vol = None
        self.Weighted_Mean_Vol = None
        self.IsRebalanced = None


def adjust_weights_based_on_yield_curve(mdp_port):
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

    # Rising Inflation, Growth and Flattening, Positive YC - Commodities

    # Falling Growth, High Inflation And Negative And Falling YC - Cash

    # Low Growth, Low Inflation, Rising, Negative YC, Or Steeply Rising and Positive YC - Bonds

    # Higher Growth, Low Inflation, Positive YC That is not Changing Slope Too Quickly - Stocks

    return mdp_port

