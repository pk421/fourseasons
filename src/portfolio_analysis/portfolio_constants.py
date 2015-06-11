__author__ = 'Mike'


# Brute Force Calcs = # nodes ^ # assets
custom_assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
# custom_assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD', 'JNK', 'RSX', 'ILF']
# custom_assets_list = ['TNA', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'SLV']
# custom_assets_list = ['VTI', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
# custom_assets_list = ['SPXL', 'TYD', 'DRN', 'DGP', 'EDC']

# Setting IEF simulates a 3x fund, use TYD to actually get 3x, TLT has more data and TMF (3x) is more liquid, it works well
# custom_assets_list = ['VTI', 'TYD', 'DRN', 'VWO', 'DGP'] # Leveraged Version
### custom_assets_list = ['IWM', 'EFA', 'VWO', 'GLD', 'VNQ', 'TLT']
## custom_assets_list = ['IWM', 'TLT', 'GLD']
# custom_assets_list = ['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
# custom_assets_list = ['SPY', 'TLT', 'GLD', 'DBC']

# Dalio's All Weather
dalio_all_weather = ['SPY', 'IEF', 'TLT', 'GLD', 'DBC']

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


spy_sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'IBB']
countries_regions = ['EWJ', 'EFA', 'ILF', 'EEM', 'IYR', 'RWX', 'VWO']
other = ['VNQ']
us_stocks = ['SPY', 'IWM', 'SDY', 'DIA']
safe_havens = ['TLT', 'IEF', 'AGG']
commodities = ['DBC', 'GLD', 'SLV']


full_list_of_possibilities = spy_sectors + countries_regions + other + us_stocks + safe_havens + commodities


custom_assets_list = full_list_of_possibilities


##########################

m_ib = [('TNA',473), ('TMF', 809), ('TLT',491), ('IEF', 528),('UCD', 206), ('GLD', 270)]
vault_cash = 580.87

# roth_ira = [('TNA',106), ('DRN',4), ('TMF',176), ('EURL',336), ('UGLD',180)]
roth_ira = [('TNA',132), ('TMF',106), ('DBC', 836), ('UGLD',337), ('EURL', 0), ('EDC', 0), ('DRN', 0)]
roth_401k = [('VIIIX',44.176), ('VTPSX',13.703),  ('VBMPX',2357.188)]

#####['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
c_ib = [('TNA',1455), ('TLT',3488), ('IEF', 1500), ('DBC',4365), ('UGLD',2612)]
# c_roth_ira = [('TNA',190), ('DRN',7), ('TMF',315), ('EURL',605), ('UGLD',327)]
c_roth_ira = [('TNA',237), ('TMF',190), ('DBC',1503), ('UGLD',605), ('EURL', 0), ('EDC', 0), ('DRN', 0)]
c_trad_ira = [('SDY',0), ('TLT',0), ('DBC',0), ('GLD',0), ('EFA', 0), ('VWO', 0), ('VNQ', 0)]

live_portfolio = [c_trad_ira]