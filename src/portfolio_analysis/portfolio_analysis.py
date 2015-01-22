import numpy as np
import scipy as scipy
import datetime

import math

import src.toolsx as tools

from src.indicator_system                           import get_sharpe_ratio
from src.portfolio_analysis.portfolio_utils         import get_data
from src.math_tools                                 import get_returns, get_ln_returns

import logging
logging.root.setLevel(logging.INFO)

def do_optimization(mdp_port, x):
    theoretical_weights = mdp_port.get_tangency_weights(x)

    # theoretical_weights = np.array([ [1.0 / len(mdp_port.assets)] for x in mdp_port.assets])
    # theoretical_weights = np.array([ [0.30], [0.15], [0.40], [0.075], [0.075] ])
    # theoretical_weights = np.array([ [0.30], [0.55], [0.15] ])

    print "Theoretical Result: \n", theoretical_weights
    print "weighted vols equal one check: ", mdp_port.weighted_vols_equal_one(theoretical_weights)

    # This section should not be necessary since weights are already normalized. It's an added safety.
    abs_weight_sum = np.sum([abs(n) for n in theoretical_weights])
    real_weight_sum = np.sum(theoretical_weights)
    weight_ratio = real_weight_sum / abs_weight_sum if round(abs_weight_sum, 6) != 1.0 else 1.0

    normalized_theoretical_weights = np.array([ n * weight_ratio for n in theoretical_weights])

    return mdp_port, normalized_theoretical_weights

def run_portfolio_analysis():

    # Brute Force Calcs = # nodes ^ # assets
    # assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
    # assets_list = ['TNA', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'SLV']
    # assets_list = ['VTI', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
    # assets_list = ['SPXL', 'TYD', 'DRN', 'DGP', 'EDC']

    # Setting IEF simulates a 3x fund, use TYD to actually get 3x, TLT has more data and TMF (3x) is more liquid, it works well
    # assets_list = ['VTI', 'TYD', 'DRN', 'VWO', 'DGP'] # Leveraged Version
    assets_list = ['IWM', 'EFA', 'VWO', 'GLD', 'VNQ', 'TLT']
    ## assets_list = ['IWM', 'TLT', 'GLD']
    # assets_list = ['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
    # assets_list = ['SPY', 'IEF', 'TLT', 'DBC', 'GLD']

    # Dalio's All Weather
    # assets_list = ['SPY', 'IEF', 'TLT', 'GLD', 'DBC']

    # Dividend Payers:
    # assets_list = ['T', 'MCD', 'CVX', 'TGT', 'KO', 'PG', 'KMB', 'CINF', 'MMM']

    # assets_list = ['VNQ', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'IBB', 'TLT', 'GLD']
    # assets_list = ['VNQ', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'IBB', 'TLT', 'GLD']
    # assets_list = ['TNA', 'TLT', 'GLD', 'UCD', 'FAS', 'TECL', 'YINN', 'LBJ', 'TQQQ', 'CURE', 'UDOW']

    # TSP: G | C, F, S, I - (consider the G fund cash since it can't go down)
    # assets_list = ['SPY', 'AGG', 'FSEMX', 'EFA']

    # IRA: Fidelity Commission Free:
    # S&P 500, US Small Cap, Short Term Treasury Bonds, Total US Bond Market, Dow Jones Real Estate, EAFE, BRICS, Emerging Markets, Gold Miners
    # assets_list = ['IVV', 'IJR', 'SHY', 'AGG', 'IYR', 'IEFA', 'BKF', 'IEMG', 'RING']
    # assets_list = ['IYR', 'IEFA', 'IEMG']
    # assets_list = ['VTI', 'IEF', 'VNQ', 'EEM']

    # 401k
    # Note: VBMPX has a shorter duration and is actually less volatile than VIPIX, lowers returns slightly but greatly boosts Sharpe
    # VEMPX is small caps basically but has not done as well in this portfolio compared to VIIIX
    # The five asset list here is less volatile and better diversified, but has a lower return
    # VIIIX=SPY, VBMPX=AGG, VTPSX=VEU
    # assets_list = ['VIIIX', 'VEMPX', 'VTPSX', 'VIPIX', 'VBMPX']
    # assets_list = ['VIIIX', 'VEU', 'LAG']
    # assets_list = ['VIIIX', 'VTPSX', 'VIPSX']
    # assets_list = ['SPY', 'VEU', 'AGG']

    mdp_port = MDPPortfolio(assets_list=assets_list)
    logging.debug(str(mdp_port.assets))
    mdp_port = get_data(mdp_port, base_etf=mdp_port.assets[0], last_x_days=0, get_new_data=True)

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

    rebalance_log = []
    while x < len(mdp_port.trimmed[mdp_port.assets[0]]):
        if not mdp_port.rebalance_now:

            mdp_port.x = x
            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)
            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation])
            print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation
            mdp_port.rebalance_counter += 1

            if x >= mdp_port.rebalance_time:
                # The statement below DOES have an impact on whether a rebalance is hit in this if block
                # It also affects overall program flow and the end result
                _ = mdp_port.get_tangency_weights(x)
                trailing_diversification_ratio = mdp_port.get_diversification_ratio(weights='current')
                mdp_port.trailing_DRs.append(trailing_diversification_ratio)
                rebalance_date = mdp_port.trimmed[mdp_port.assets[0]][x]['Date']
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
            print "\n\n\nTheoretical Weights: ", str(theoretical_weights)
            mdp_port.set_normalized_weights(theoretical_weights, old_weighted_valuation, x)
            ###

            # calling get_port_valuation here is used to set the current weights, used in the div ratio below
            _ = get_port_valuation(mdp_port, x=x)
            for k, asset in enumerate(mdp_port.assets):
                if round(mdp_port.normalized_weights[k], 6) != round(mdp_port.current_weights[k], 6):
                    raise Exception("Normalized weights do not equal the current weights after setting normalized.")

            trailing_diversification_ratio = mdp_port.get_diversification_ratio(weights='normalized')
            mdp_port.trailing_DRs.append(trailing_diversification_ratio)
            print "New Trailing Diversification Ratio: ", x, trailing_diversification_ratio
            rebalance_log.append((rebalance_date, rebalance_old_div_ratio, trailing_diversification_ratio, theoretical_weights))

            current_portfolio_valuation = get_port_valuation(mdp_port, x=x)

            if round(current_portfolio_valuation, 6) != round(old_weighted_valuation, 6):
                raise Exception("Current Valuation != Old Weighted Valuation: Curr/Old: " + str(current_portfolio_valuation) + '\t' + str(old_weighted_valuation))

            print "Current Portfolio New Valuation: ", current_portfolio_valuation

            mdp_port.portfolio_valuations.append([mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation ])
            print x, mdp_port.trimmed[mdp_port.assets[0]][x]['Date'], current_portfolio_valuation
            x+=1

    print '\n'.join([str(r) for r in rebalance_log])

    for i, p in enumerate(mdp_port.portfolio_valuations):
        if i == 0:
            starting_value = p[1]
        mdp_port.portfolio_valuations[i][1] = mdp_port.portfolio_valuations[i][1] / starting_value


    if len(mdp_port.trimmed[mdp_port.assets[0]]) != len(mdp_port.portfolio_valuations):
        raise Exception('Length of asset list data does not match length of valuations data.')

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


    print '\n', mdp_port.portfolio_valuations[0], mdp_port.portfolio_valuations[-1]

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


    output_string = '\n'.join( [(str(n[0]) + ',' + str(n[1]) + ',' + str(ref_log[k][2]/ref_log[0][2]) +\
                                 ',' + str(system_drawdown[k]) + ',' + str(ref_drawdown[k]) +\
                                 ',' + str(system_dma[k]) +\
                                 ',' + str(mdp_port.trailing_DRs[k])
        ) for k, n in enumerate(mdp_port.portfolio_valuations)] )

    current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_file_name = '/home/wilmott/Desktop/fourseasons/fourseasons/results/portfolio_analysis' + '_' + str(current_time) +'.csv'
    with open(out_file_name, 'w') as f:
        f.write(output_string)

    total_years_in = len(sharpe_price_list) / 252.0
    system_annualized_return = math.pow(mdp_port.portfolio_valuations[-1][1], (1.0/total_years_in))

    # We added leading zeroes to trailing_DRs, don't include them in the mean
    print "Mean Diversification Ratio: ", np.mean([ n for n in mdp_port.trailing_DRs if n != 0.0 ]), len(rebalance_log)
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

        C = np.dot(np.dot(np.matrix.transpose(volatilities_matrix), scipy.linalg.inv(self.cov_matrix)), volatilities_matrix)
        # print "C is: ", C

        #D = (A * C) - (B^2)
        #print "D is: ", D

        numerator = np.dot(scipy.linalg.inv(self.cov_matrix), volatilities_matrix)
        denominator = B
        tangency_weights = np.divide(numerator, denominator)

        total_sum = sum([abs(n) for n in tangency_weights])[0]
        normalized_weights = np.divide(tangency_weights, total_sum)

        return normalized_weights


    def set_normalized_weights(self, weights, old_weighted_valuation, x):
        self.normalized_weights = weights
        weight_sum = np.sum([abs(n) for n in weights])
        if round(weight_sum, 6) != 1.0:
            raise Exception("set normalized weights don't sum to 1.0: " + str(weight_sum))
        self.current_entry_prices = [ self.closes[v][x] for k, v in enumerate(self.assets)]
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


def run_live_portfolio_analysis():

    vault = [('TNA',473), ('TMF',809), ('UCD',655), ('UGLD',804), ('EURL', 336)]
    vault_cash = 580.87

    roth_ira = [('TNA',106), ('DRN',4), ('TMF',176), ('EURL',336), ('UGLD',180)]
    roth_401k = [('VIIIX',44.176), ('VTPSX',13.703),  ('VBMPX',2357.188)]

    #####['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
    designated_benificiary = [('TNA',1455), ('TMF',1767), ('UCD',3695), ('UGLD',2099)]
    c_roth_ira = [('TNA',190), ('DRN',7), ('TMF',315), ('EURL',605), ('UGLD',327)]

    all_portfolios = [c_roth_ira]

    for account in all_portfolios:
        cash = 0

        assets = [ n[0] for n in account ]
        shares = [ n[1] for n in account ]

        port = MDPPortfolio(assets)

        port = get_data(port, base_etf=port.assets[0], last_x_days=0, get_new_data=True)

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
        div_ratio = port.get_diversification_ratio(weights='current')

        print "Assets: \t", assets
        print "Tangency: \t", [ round(n[0], 6) for n in tangency_weights ]
        print "Act. Weights: \t", [ round(n[0], 6) for n in port.current_weights ]
        print "Value: \t\t", valuation
        print "Div Ratio: \t", div_ratio

    return


