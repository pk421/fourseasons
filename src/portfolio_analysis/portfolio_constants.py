__author__ = 'Mike'


# Brute Force Calcs = # nodes ^ # assets
# custom_assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
# custom_assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD', 'JNK', 'RSX', 'ILF']
# custom_assets_list = ['TNA', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'SLV']
# custom_assets_list = ['VTI', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
# custom_assets_list = ['SPXL', 'TYD', 'DRN', 'DGP', 'EDC']

# Setting IEF simulates a 3x fund, use TYD to actually get 3x, TLT has more data and TMF (3x) is more liquid, it works well
# custom_assets_list = ['VTI', 'TYD', 'DRN', 'VWO', 'DGP'] # Leveraged Version
custom_assets_list = ['IWM', 'EFA', 'VWO', 'GLD', 'VNQ', 'TLT']
## custom_assets_list = ['IWM', 'TLT', 'GLD']
# custom_assets_list = ['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
# custom_assets_list = ['SPY', 'TLT', 'GLD', 'DBC']

# Dalio's All Weather
# custom_assets_list = ['SDY', 'IEF', 'TLT', 'GLD', 'DBC']

# Dividend Payers:
# custom_assets_list = ['T', 'MCD', 'CVX', 'TGT', 'KO', 'PG', 'KMB', 'CINF', 'MMM']

# custom_assets_list = ['VNQ', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'IBB', 'TLT', 'GLD']
# custom_assets_list = ['VNQ', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'IBB', 'TLT', 'GLD']
# custom_assets_list = ['TNA', 'TLT', 'GLD', 'UCD', 'FAS', 'TECL', 'YINN', 'LBJ', 'TQQQ', 'CURE', 'UDOW']

# TSP: G | C, F, S, I - (consider the G fund cash since it can't go down)
# custom_assets_list = ['SPY', 'AGG', 'FSEMX', 'EFA']

# IRA: Fidelity Commission Free:
# S&P 500, US Small Cap, Short Term Treasury Bonds, Total US Bond Market, Dow Jones Real Estate, EAFE, BRICS, Emerging Markets, Gold Miners
# custom_assets_list = ['IVV', 'IJR', 'SHY', 'AGG', 'IYR', 'IEFA', 'BKF', 'IEMG', 'RING']
# custom_assets_list = ['IYR', 'IEFA', 'IEMG']
# custom_assets_list = ['VTI', 'IEF', 'VNQ', 'EEM']

# 401k
# Note: VBMPX has a shorter duration and is actually less volatile than VIPIX, lowers returns slightly but greatly boosts Sharpe
# VEMPX is small caps basically but has not done as well in this portfolio compared to VIIIX
# The five asset list here is less volatile and better diversified, but has a lower return
# VIIIX=SPY, VBMPX=AGG, VTPSX=VEU
# custom_assets_list = ['VIIIX', 'VEMPX', 'VTPSX', 'VIPIX', 'VBMPX']
# custom_assets_list = ['VIIIX', 'VEU', 'LAG']
# custom_assets_list = ['VIIIX', 'VTPSX', 'VIPSX']
# custom_assets_list = ['SPY', 'VEU', 'AGG']


##########################

vault = [('TNA',473), ('TMF',809), ('UCD',825), ('UGLD',890)]
vault_cash = 580.87

# roth_ira = [('TNA',106), ('DRN',4), ('TMF',176), ('EURL',336), ('UGLD',180)]
roth_ira = [('TNA',132), ('TMF',106), ('DBC', 836), ('UGLD',337)]
roth_401k = [('VIIIX',44.176), ('VTPSX',13.703),  ('VBMPX',2357.188)]

#####['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
designated_benificiary = [('TNA',1455), ('TMF',2017), ('UCD',3695), ('UGLD',2612)]
# c_roth_ira = [('TNA',190), ('DRN',7), ('TMF',315), ('EURL',605), ('UGLD',327)]
c_roth_ira = [('TNA',237), ('TMF',190), ('DBC',1503), ('UGLD',605)]

live_portfolio = [c_roth_ira]