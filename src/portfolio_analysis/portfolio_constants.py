__author__ = 'Mike'


# Brute Force Calcs = # nodes ^ # assets
original_mdp = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
# custom_assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD', 'JNK', 'RSX', 'ILF']
# custom_assets_list = ['TNA', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'SLV']
# custom_assets_list = ['VTI', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
# custom_assets_list = ['SPXL', 'TYD', 'DRN', 'DGP', 'EDC']

# Setting IEF simulates a 3x fund, use TYD to actually get 3x, TLT has more data and TMF (3x) is more liquid, it works well
# custom_assets_list = ['VTI', 'TYD', 'DRN', 'VWO', 'DGP'] # Leveraged Version
### custom_assets_list = ['IWM', 'EFA', 'VWO', 'GLD', 'VNQ', 'TLT']
## custom_assets_list = ['IWM', 'TLT', 'GLD']
custom_assets_list = ['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
# custom_assets_list = ['SPY', 'TLT', 'GLD', 'DBC']

# Dalio's All Weather
# dalio_all_weather = ['IJR', 'IEF', 'TLT', 'GLD', 'DJP', 'XLE']
dalio_all_weather = ['IJR', 'IEF', 'TLT', 'GLD', 'DBC']
# dalio_all_weather = ['VSMAX', 'VFITX', 'VUSTX']
modified_all_weather = ['IWM', 'IEF', 'TLT', 'GLD', 'XLE']

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
# custom_assets_list = ['VIIIX', 'VBMPX', 'VTPSX']
# custom_assets_list = ['SPY', 'VEU', 'AGG']


spy_sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'IBB']
countries_regions = ['EWJ', 'EFA', 'ILF', 'FXI', 'EEM', 'RWX', 'VWO']
other = ['VNQ', 'IYR']
us_stocks = ['SPY', 'IWM', 'SDY', 'DIA', 'QQQ']
safe_havens = ['TLT', 'IEF', 'AGG']
commodities = ['DBC', 'GLD', 'SLV']

# 2015-12-14
modified_mdp = ['VTI', 'TLT', 'EMB', 'GLD', 'DBC']
modified_mdp_2 = ['VTI', 'TLT', 'IEF', 'EMB', 'LQD', 'GLD', 'DBC']


full_list_of_possibilities = spy_sectors + countries_regions + other + us_stocks + safe_havens + commodities

top_10_sdy = ['T', 'HCP', 'PBCT', 'ED', 'O', 'NNN', 'SYY', 'MCD', 'KMB', 'CVX']

### custom_assets_list = countries_regions

# hedged_dividends = ['CVX', 'HCP', 'T', 'ERY', 'DRV']
hedged_dividends = ['IWM', 'TLT', 'IEF', 'UCD', 'GLD', 'PAA']
hedged_dividends = ['IWM', 'TLT', 'IEF', 'UCD', 'GLD', 'PAA', 'T', 'HCP', 'OKE']
# hedged_dividends = ['HCP', 'T', 'HP', 'CVX', 'MCY', 'ORI', 'STR', 'BKH', 'EMR', 'NWN', 'MO', 'UVV', 'BRC', 'XOM', 'ED',\
#                     'NUE', 'DBD', 'PG', 'VVC', 'SON', 'CTBI', 'MCD', 'CINF', 'PNY', 'KO', 'WGL', 'KMB', 'MSEX',\
#                     'JNJ', 'TMP', 'CWT', 'RAVN', 'SYY', 'NFG', 'PEP', 'WMT', 'CTWS', 'WEYS', 'TROW', 'LEG']

custom_assets_list = dalio_all_weather


##########################

live_mdp = [('SPY', 1), ('EFA', 1), ('EWJ', 1), ('EEM', 1), ('IYR', 1), ('RWX', 1), ('IEF', 1), ('TLT', 1), ('DBC', 1), ('GLD', 1)]
spy_sectors = [('XLF', 1), ('XLK', 1), ('XLE', 1), ('XLV', 1), ('XLI', 1), ('XLP', 1), ('XLU', 1), ('XLY', 1), ('XLB', 1), ('IBB', 1)]
countries_regions = [('EWJ', 1), ('EFA', 1), ('ILF', 1), ('FXI', 1), ('EEM', 1), ('RWX', 1), ('VWO', 1)]
hedges = [('TLT', 1), ('IEF', 1), ('GLD', 1)]

# m_ib = [('IWM',932), ('TLT',1600), ('IEF', 528), ('UCD', 206), ('GLD', 270), ('PAA', 900)]
# m_ib = [('IWM',932), ('TLT',1600), ('IEF', 528), ('UCD', 206), ('GLD', 270)]

m_ib = [('IJR', 1100), ('TLT', 1601), ('IEF', 600), ('UCD', 500), ('IAU', 3000), ('PAA', 900)]
# m_ib = [('IJR', 100), ('SPY', 100) ]
m_ib = [('IJR', 1100), ('TLT', 1601), ('IEF', 600), ('UCD', 500), ('IAU', 3000), ('PAA', 900), ('EV', 2000), ('T', 1200), ('STR', 1800) ]
# m_ib = [('IJR', 1100), ('TLT', 1601), ('IEF', 600), ('UCD', 500), ('IAU', 3000), ('PAA', 900), ('ED', 1200), ('HCP', 500) ]
# m_ib = [('IWM', 1050), ('TLT', 1600), ('IEF', 600), ('UCD', 500), ('IAU', 3000), ('PAA', 900), ('T', 500), ('HCP', 500)]

# roth_ira = [('TLT',122.7), ('DBC', 360), ('UGLD',185), ('VGK', 203.9)]
roth_ira = [('TLT',123.27), ('DBC', 360), ('UGLD',185), ('VGK', 304.7), ('IWM', 0)]

roth_401k = [('VIIIX',44.176), ('VTPSX',13.703),  ('VBMPX',2357.188)]

#####['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
# c_ib = [('IWM',2835), ('TLT',3488), ('IEF', 1500), ('DBC',4365), ('UGLD',2612), ('PAA', 1850)]
c_ib = [('IWM', 2835), ('TLT', 3488), ('IEF', 1500), ('DBC', 5800), ('UGLD', 3400), ('PAA', 1850)]

c_roth_ira = [('TLT', 775), ('DBC', 2600), ('UGLD', 1350), ('VGK', 1500)]
# c_roth_ira = [('TLT',190), ('DBC',1503), ('UGLD',605), ('VGK', 0)]

stocks_to_test = [ ('HCP', 100), ('T', 100), ('ED', 100), ('UHT', 100), ('MCY', 100), ('HP', 100), ('CVX', 100), \
                   ('ORI', 100), ('UVV', 100), ('ABBV', 100), ('NWN', 100), ('STR', 100), ('EMR', 100), ('CTBI', 100), \
                   ('XOM', 100), ('PG', 100), ('VVC', 100), ('RAVN', 100), ('KO', 100), ('SON', 100), ('EV', 100), \
                   ('BRC', 100), ('NUE', 100), ('ADM', 100), ('NFG', 100), ('KMB', 100), ('WGL', 100), ('CINF', 100), \
                   ('WEYS', 100), ('TROW', 100), ('JNJ', 100), ('WMT', 100), ('BKH', 100), ('TGT', 100), ('PEP', 100), \
                   ('THFF', 100), ('GPC', 100), ('SYY', 100), ('LEG', 100), ('MMM', 100), ('MCD', 100), ('MSA', 100), \
                   ('ABT', 100), ('DOV', 100), ('TMP', 100), ('CWT', 100), ('FRT', 100), ('SJW', 100), ('ADP', 100), \
                  ]


live_portfolio = [m_ib]