__author__ = 'Mike'


# Brute Force Calcs = # nodes ^ # assets
original_mdp = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
# custom_assets_list = ['SPY', 'EFA', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD', 'JNK', 'RSX', 'ILF']
# custom_assets_list = ['VTI', 'EFA', 'EWJ', 'EEM', 'VNQ', 'RWX', 'IEF', 'TLT', 'DBC', 'GLD']
# custom_assets_list = ['SPXL', 'TYD', 'DRN', 'DGP', 'EDC']

# Setting IEF simulates a 3x fund, use TYD to actually get 3x, TLT has more data and TMF (3x) is more liquid, it works well
# custom_assets_list = ['VTI', 'TYD', 'DRN', 'VWO', 'DGP'] # Leveraged Version
### custom_assets_list = ['IWM', 'EFA', 'VWO', 'GLD', 'VNQ', 'TLT']
## custom_assets_list = ['IWM', 'TLT', 'GLD']
# custom_assets_list = ['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
# custom_assets_list = ['SPY', 'TLT', 'GLD', 'DBC']

# Dalio's All Weather
# dalio_all_weather = ['IJR', 'IEF', 'TLT', 'GLD', 'DJP', 'XLE']
# dalio_all_weather = ['VBR', 'IEF', 'TLT', 'GLD', 'DBC', 'BSV']
dalio_all_weather = ['VBR', 'IEF', 'TLT', 'GLD', 'DBC']

em_all_weather = ['VWO', 'IEF', 'VWOB', 'GLD', 'DBC']

# dalio_all_weather = ['VSMAX', 'VFITX', 'VUSTX']
modified_all_weather = ['VBR', 'IEF', 'TLT', 'GLD', 'DBC', 'VWO', 'PCY'] # VWOB is better, but PCY has more history

# From PortfolioCharts.com
# golden_butterfly = ['VBR', 'SPY', 'SHY', 'TLT', 'GLD', 'BSV']
golden_butterfly = ['SPY', 'VBR', 'TLT', 'SHY', 'GLD']
modified_butterfly = ['VBR', 'VWO', 'TLT', 'VWOB', 'BSV', 'GLD']


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


# XLRE is the "sector fund" for real estate, but VNQ is the Vanguard one
spy_sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'VNQ']
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
hedged_dividends = ['IWM', 'TLT', 'IEF', 'DBC', 'GLD', 'PAA']
hedged_dividends = ['IWM', 'TLT', 'IEF', 'DBC', 'GLD', 'PAA', 'T', 'HCP', 'OKE']
# hedged_dividends = ['HCP', 'T', 'HP', 'CVX', 'MCY', 'ORI', 'STR', 'BKH', 'EMR', 'NWN', 'MO', 'UVV', 'BRC', 'XOM', 'ED',\
#                     'NUE', 'DBD', 'PG', 'VVC', 'SON', 'CTBI', 'MCD', 'CINF', 'PNY', 'KO', 'WGL', 'KMB', 'MSEX',\
#                     'JNJ', 'TMP', 'CWT', 'RAVN', 'SYY', 'NFG', 'PEP', 'WMT', 'CTWS', 'WEYS', 'TROW', 'LEG']

max_diversity = [ 'SPY', 'TLT', 'IEF', 'GLD', 'DBC', 'PCY', 'VWO', 'RWO', 'MUB']
# Adding GBTC significantly reduces the history we can test against
# max_diversity = [ 'SPY', 'TLT', 'IEF', 'GLD', 'DBC', 'PCY', 'VWO', 'RWO', 'GBTC']
# max diversity: also add MUB, VTEB - muni bonds

custom_assets_list = max_diversity


##########################
# manually optimized in latest 63 days
max_uncorrelated = [('SPY', 1), ('TLT', 1), ('IEF', 1), ('GLD', 1), ('DBC', 4), ('PCY', 0.000001), ('VWO', 2), \
                    ('RWO', 0), ('MUB', 3), ('GBTC', 1),
                    ]

m_ib_all_assets = [
       # Developed Market stocks
       ('T', 1400), ('VOO', 0.001), ('VBR', 700),

       # Developed Market Bonds
       ('VGLT', 0.001), ('VGIT', 0.0001),

       # Developed Market Muni Bonds
       ('MUB', 0.0001),

       # Emerging Market Bonds
       ('VWOB', 0.001),

       # Emerging Market Stocks
       ('EWZ', 1000), ('GXG', 4000), ('RSX', 2000), ('VWO', 0.0001),

       # Commodities
       # Switch DBC to PDBC
       ('DBC', 4000), ('HP', 500), ('MPLX', 2000), ('PAA', 1000),

       # Gold
       ('GDX', 3800), ('IAU', 7000),

       # Real Estate
       ('VNQ', 0.0001),

       # Crypto
       ('GBTC', 0.001),

       # Other
       ('ACB', 5506), ('APHA', 1333), ('HMLSF', 1333), ('LXRP', 5985), ('CGC', 798),

       ]


m_ib_20191215 = [
       # Developed Market stocks
       ('T', 1400), ('VOO', 200), ('VBR', 700), ('SKT', 2000),

       # Developed Market Bonds
       ('VGLT', 0.0001), ('VGIT', 0.0001), ('EDV', 0.0001),

       # Developed Market Muni Bonds
       ('MUB', 0.0001),

       # Emerging Market Bonds
       ('VWOB', 0.0001),

       # Emerging Market Stocks
       ('EWZ', 1000), ('GXG', 4000), ('RSX', 2000), ('VWO', 0.0001),

       # Commodities
       # Switch DBC to PDBC
       ('DBC', 4000), ('HP', 600), ('MPLX', 0.0001), ('PAA', 1000),

       # Gold
       ('GDX', 3800), ('IAU', 7000),

       # Real Estate
       ('VNQ', 300),

       # Crypto
       ('GBTC', 0.0001),

       # Other
       ('ACB', 0.0001), ('APHA', 0.0001), ('HMLSF', 1333), ('LXRP', 5985), ('CGC', 400),

       ]


# m_ib = max_uncorrelated
# m_ib = m_ib_all_assets

m_golden_butterfly_20191020 = [('IAU', 84), ('VGLT', 14), ('VGSH', 20), ('VBR', 9), ('VTI', 8),

       ]
# m_ib_total = [('IJR', 2200), ('TLT', 1866), ('IEF', 600), ('DBC', 4000), ('IAU', 5000), ('GDX', 500), ('PAA', 900), ('GXG', 4000), ('VNQ', 0), ('PCY', 0), ('JNK', 0), ('VIPSX', 0), ('XLU', 0)]
#
# m_ib_pot = [('IJR', 2100), ('TLT', 1700), ('IEF', 0), ('DBC', 2000), ('IAU', 0), ('GDX', 500), ('PAA', 0), ('GXG', 4000),
#             ('HP', 500), ('CANN', 970), ('HEMP', 49000), ('CBDS', 280), ('LXRP', 4500), ('MJNA', 4000), ('CBIS', 10000),
#             ('MCIG', 3200), ('CNAB', 600), ('AMMJ', 1000),
#
#             # these are tricky because they don't use the same symbols as IB and the quantities must be adjusted for the
#             # exchange rate / the change in symbol
#             ('ACBFF', 2459), ('TWMJF', 766)
#             # ('ACB', 2000), ('APH', 1000), ('LEAF', 600), ('WEED', 600)
#             ]

# EM Strat
# em_strat = [('FXI', 100), ('RSX', 300), ('EWZ', 100), ('EMB', 100)]


roth_ira = [('TLT', 128), ('DBC', 560), ('IAU',573), ('IJR', 206), ('XOP', 140)]

#####['TNA', 'EURL', 'EDC', 'UGLD', 'DRN', 'TMF']
### c_ib = [('IWM', 2835), ('TLT', 3488), ('IEF', 1500), ('DBC', 5800), ('UGLD', 3400), ('PAA', 1850)]
c_ib = [('VBR', 3240), ('TLT', 3488), ('VGLT', 1660), ('IEF', 1970), ('DBC', 6870), ('IAU', 11000), ('PAA', 1850)]

c_roth_ira = [('TLT', 775), ('DBC', 2600), ('UGLD', 1350), ('VGK', 1500)]
# c_roth_ira = [('TLT',190), ('DBC',1503), ('UGLD',605), ('VGK', 0)]

# Developed World Large Cap Stocks - Dividend.com 25 Year Dividend Stocks - incomplete list
dividend_growers = [
    ('ABBV', 0.000001),
    ('ABT', 0.000001),
    ('ADM', 0.000001),
    ('ADP', 0.000001),
    ('BKH', 0.000001),
    ('BRC', 0.000001),
    ('CINF', 0.000001),
    ('CTBI', 0.000001),
    ('CVX', 0.000001),
    ('CWT', 0.000001),
    ('DOV', 0.000001),
    ('ED', 0.000001),
    ('EMR', 0.000001),
    ('EV', 0.000001),
    ('FRT', 0.000001),
    ('GPC', 0.000001),
    ('HCP', 0.000001),
    ('HP', 0.000001),
    ('JNJ', 0.000001),
    ('KMB', 0.000001),
    ('KO', 0.000001),
    ('LEG', 0.000001),
    ('MCD', 0.000001),
    ('MCY', 0.000001),
    ('MMM', 0.000001),
    ('MSA', 0.000001),
    ('NFG', 0.000001),
    ('NUE', 0.000001),
    ('NWN', 0.000001),
    ('ORI', 0.000001),
    ('PEP', 0.000001),
    ('PG', 0.000001),
    ('RAVN', 0.000001),
    ('SKT', 0.000001),
    ('SJW', 0.000001),
    ('SON', 0.000001),
    ('SYY', 0.000001),
    ('T', 0.000001),
    ('TGT', 0.000001),
    ('THFF', 0.000001),
    ('TMP', 0.000001),
    ('TROW', 0.000001),
    ('UHT', 0.000001),
    ('UVV', 0.000001),
    ('WEYS', 0.000001),
    ('WMT', 0.000001),
    ('XOM', 0.000001),
    ]

# Developed World ETFS
developed_world_etfs = [
    ('EFA', 0.000001),
    ('EWA', 0.000001),
    ('EWJ', 0.000001),
]

# USA Large Indices
usa_large_indices = [
    ('IJR', 0.000001),
    ('SPY', 0.000001),
    ('VBR', 0.000001),
    ('VOO', 0.000001),
    ('VTI', 0.000001),
]

# Emerging Markets
emerging_market_etfs = [
    ('EEM', 0.000001),
    ('EWZ', 0.000001),
    ('GXG', 0.000001),
    ('ICOL', 0.000001),
    ('RSX', 0.000001),
]

# Developed Market Bonds
developed_world_bonds = [
    ('EDV', 0.000001),
    ('MUB', 0.000001),
    ('SCHP', 0.000001),
    ('SHY', 0.000001),
    ('VGIT', 0.000001),
    ('VGLT', 0.000001),
]

# Emerging Market Bonds
emerging_market_bonds = [
    ('EMB', 0.000001),
    ('PCY', 0.000001),
    ('VWOB', 0.000001),
]

# REITs
reits = [
    ('IYR', 0.000001),
    ('RWX', 0.000001),
    ('VNQ', 0.000001),
]

# Commodities
commodities = [
    ('DBC', 0.000001),
    ('MPLX', 0.000001),
    ('PAA', 0.000001),
]

# Precious Metals
precious_metals = [
    ('GDX', 0.000001),
    ('GDXJ', 0.000001),
    ('IAU', 0.000001),
    ('SLV', 0.000001),
    ('USV', 0.000001),
]

# Sector ETFs
sector_etfs = [
    ('IBB', 0.000001),
    ('XLB', 0.000001),
    ('XLE', 0.000001),
    ('XLF', 0.000001),
    ('XLI', 0.000001),
    ('XLK', 0.000001),
    ('XLP', 0.000001),
    ('XLU', 0.000001),
    ('XLV', 0.000001),
    ('XLY', 0.000001),
]

# Marijuana
marijuana = [
    ('ACB', 0.000001),
    ('APHA', 0.000001),
    ('CGC', 0.000001),
    ('HMLSF', 0.000001),
    ('LXRP', 0.000001),
]

# Crypto
cryptocurrencies = [
    ('GBTC', 0.000001),
]

stocks_to_test = dividend_growers + developed_world_etfs + usa_large_indices + emerging_market_etfs + \
                      developed_world_bonds + emerging_market_bonds + reits + commodities + precious_metals + \
                      sector_etfs + marijuana + cryptocurrencies

# with open('/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + '300B_1M_and_etfs_etns.csv', 'r') as file:
# with open('/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + 'big_etfs.csv', 'r') as file:
#     tickers = file.read().split('\n')
# file_stocks = [ (s.replace('\r', ''), 1.0) for s in tickers if s.replace('\r', '') not in [ a[0] for a in stocks_to_test] ]
# stocks_to_test = stocks_to_test + file_stocks

# full_stocks_to_test = dividend_growers + developed_world_etfs + usa_large_indices + emerging_market_etfs + \
#                       developed_world_bonds + emerging_market_bonds + reits + commodities + precious_metals + \
#                       sector_etfs + marijuana + cryptocurrencies + file_stocks

###
# original_mdp = [('SPY', 100), ('EFA', 100), ('EWJ', 100), ('EEM', 0), ('IYR', 0), ('RWX', 0), ('IEF', 100), \
#                 ('TLT', 100), ('DBC', 100), ('GLD', 100)]
###

live_portfolio = [stocks_to_test]




# S&P 500 Components
# ('MMM', 100),
# ('ABT', 100),
# ('ABBV', 100),
# ('ACN', 100),
# ('ATVI', 100),
# ('AYI', 100),
# ('ADBE', 100),
# ('AAP', 100),
# ('AES', 100),
# ('AET', 100),
# ('AMG', 100),
# ('AFL', 100),
# ('A', 100),
# ('APD', 100),
# ('AKAM', 100),
# ('ALK', 100),
# ('ALB', 100),
# ('ALXN', 100),
# ('ALLE', 100),
# ('AGN', 100),
# ('ADS', 100),
# ('LNT', 100),
# ('ALL', 100),
# ('GOOGL', 100),
# ('GOOG', 100),
# ('MO', 100),
# ('AMZN', 100),
# ('AEE', 100),
# ('AAL', 100),
# ('AEP', 100),
# ('AXP', 100),
# ('AIG', 100),
# ('AMT', 100),
# ('AWK', 100),
# ('AMP', 100),
# ('ABC', 100),
# ('AME', 100),
# ('AMGN', 100),
# ('APH', 100),
# ('APC', 100),
# ('ADI', 100),
# ('ANTM', 100),
# ('AON', 100),
# ('APA', 100),
# ('AIV', 100),
# ('AAPL', 100),
# ('AMAT', 100),
# ('ADM', 100),
# ('ARNC', 100),
# ('AJG', 100),
# ('AIZ', 100),
# ('T', 100),
# ('ADSK', 100),
# ('ADP', 100),
# ('AN', 100),
# ('AZO', 100),
# ('AVB', 100),
# ('AVY', 100),
# ('BHI', 100),
# ('BLL', 100),
# ('BAC', 100),
# ('BCR', 100),
# ('BAX', 100),
# ('BBT', 100),
# ('BDX', 100),
# ('BBBY', 100),
# ('BBY', 100),
# ('BIIB', 100),
# ('BLK', 100),
# ('HRB', 100),
# ('BA', 100),
# ('BWA', 100),
# ('BXP', 100),
# ('BSX', 100),
# ('BMY', 100),
# ('AVGO', 100),
# ('CHRW', 100),
# ('CA', 100),
# ('COG', 100),
# ('CPB', 100),
# ('COF', 100),
# ('CAH', 100),
# ('KMX', 100),
# ('CCL', 100),
# ('CAT', 100),
# ('CBOE', 100),
# ('CBG', 100),
# ('CBS', 100),
# ('CELG', 100),
# ('CNC', 100),
# ('CNP', 100),
# ('CTL', 100),
# ('CERN', 100),
# ('CF', 100),
# ('SCHW', 100),
# ('CHTR', 100),
# ('CHK', 100),
# ('CVX', 100),
# ('CMG', 100),
# ('CB', 100),
# ('CHD', 100),
# ('CI', 100),
# ('XEC', 100),
# ('CINF', 100),
# ('CTAS', 100),
# ('CSCO', 100),
# ('C', 100),
# ('CFG', 100),
# ('CTXS', 100),
# ('CME', 100),
# ('CMS', 100),
# ('COH', 100),
# ('KO', 100),
# ('CTSH', 100),
# ('CL', 100),
# ('CMCSA', 100),
# ('CMA', 100),
# ('CAG', 100),
# ('CXO', 100),
# ('COP', 100),
# ('ED', 100),
# ('STZ', 100),
# ('GLW', 100),
# ('COST', 100),
# ('COTY', 100),
# ('CCI', 100),
# ('CSRA', 100),
# ('CSX', 100),
# ('CMI', 100),
# ('CVS', 100),
# ('DHI', 100),
# ('DHR', 100),
# ('DRI', 100),
# ('DVA', 100),
# ('DE', 100),
# ('DLPH', 100),
# ('DAL', 100),
# ('XRAY', 100),
# ('DVN', 100),
# ('DLR', 100),
# ('DFS', 100),
# ('DISCA', 100),
# ('DISCK', 100),
# ('DG', 100),
# ('DLTR', 100),
# ('D', 100),
# ('DOV', 100),
# ('DOW', 100),
# ('DPS', 100),
# ('DTE', 100),
# ('DD', 100),
# ('DUK', 100),
# ('DNB', 100),
# ('ETFC', 100),
# ('EMN', 100),
# ('ETN', 100),
# ('EBAY', 100),
# ('ECL', 100),
# ('EIX', 100),
# ('EW', 100),
# ('EA', 100),
# ('EMR', 100),
# ('ETR', 100),
# ('EVHC', 100),
# ('EOG', 100),
# ('EQT', 100),
# ('EFX', 100),
# ('EQIX', 100),
# ('EQR', 100),
# ('ESS', 100),
# ('EL', 100),
# ('ES', 100),
# ('EXC', 100),
# ('EXPE', 100),
# ('EXPD', 100),
# ('ESRX', 100),
# ('EXR', 100),
# ('XOM', 100),
# ('FFIV', 100),
# ('FB', 100),
# ('FAST', 100),
# ('FRT', 100),
# ('FDX', 100),
# ('FIS', 100),
# ('FITB', 100),
# ('FSLR', 100),
# ('FE', 100),
# ('FISV', 100),
# ('FLIR', 100),
# ('FLS', 100),
# ('FLR', 100),
# ('FMC', 100),
# ('FTI', 100),
# ('FL', 100),
# ('F', 100),
# ('FTV', 100),
# ('FBHS', 100),
# ('BEN', 100),
# ('FCX', 100),
# ('FTR', 100),
# ('GPS', 100),
# ('GRMN', 100),
# ('GD', 100),
# ('GE', 100),
# ('GGP', 100),
# ('GIS', 100),
# ('GM', 100),
# ('GPC', 100),
# ('GILD', 100),
# ('GPN', 100),
# ('GS', 100),
# ('GT', 100),
# ('GWW', 100),
# ('HAL', 100),
# ('HBI', 100),
# ('HOG', 100),
# ('HAR', 100),
# ('HRS', 100),
# ('HIG', 100),
# ('HAS', 100),
# ('HCA', 100),
# ('HCP', 100),
# ('HP', 100),
# ('HSIC', 100),
# ('HES', 100),
# ('HPE', 100),
# ('HOLX', 100),
# ('HD', 100),
# ('HON', 100),
# ('HRL', 100),
# ('HST', 100),
# ('HPQ', 100),
# ('HUM', 100),
# ('HBAN', 100),
# ('IDXX', 100),
# ('ITW', 100),
# ('ILMN', 100),
# ('INCY', 100),
# ('IR', 100),
# ('INTC', 100),
# ('ICE', 100),
# ('IBM', 100),
# ('IP', 100),
# ('IPG', 100),
# ('IFF', 100),
# ('INTU', 100),
# ('ISRG', 100),
# ('IVZ', 100),
# ('IRM', 100),
# ('JBHT', 100),
# ('JEC', 100),
# ('SJM', 100),
# ('JNJ', 100),
# ('JCI', 100),
# ('JPM', 100),
# ('JNPR', 100),
# ('KSU', 100),
# ('K', 100),
# ('KEY', 100),
# ('KMB', 100),
# ('KIM', 100),
# ('KMI', 100),
# ('KLAC', 100),
# ('KSS', 100),
# ('KHC', 100),
# ('KR', 100),
# ('LB', 100),
# ('LLL', 100),
# ('LH', 100),
# ('LRCX', 100),
# ('LEG', 100),
# ('LEN', 100),
# ('LUK', 100),
# ('LVLT', 100),
# ('LLY', 100),
# ('LNC', 100),
# ('LLTC', 100),
# ('LKQ', 100),
# ('LMT', 100),
# ('L', 100),
# ('LOW', 100),
# ('LYB', 100),
# ('MTB', 100),
# ('MAC', 100),
# ('M', 100),
# ('MNK', 100),
# ('MRO', 100),
# ('MPC', 100),
# ('MAR', 100),
# ('MMC', 100),
# ('MLM', 100),
# ('MAS', 100),
# ('MA', 100),
# ('MAT', 100),
# ('MKC', 100),
# ('MCD', 100),
# ('MCK', 100),
# ('MJN', 100),
# ('MDT', 100),
# ('MRK', 100),
# ('MET', 100),
# ('MTD', 100),
# ('KORS', 100),
# ('MCHP', 100),
# ('MU', 100),
# ('MSFT', 100),
# ('MAA', 100),
# ('MHK', 100),
# ('TAP', 100),
# ('MDLZ', 100),
# ('MON', 100),
# ('MNST', 100),
# ('MCO', 100),
# ('MS', 100),
# ('MSI', 100),
# ('MUR', 100),
# ('MYL', 100),
# ('NDAQ', 100),
# ('NOV', 100),
# ('NAVI', 100),
# ('NTAP', 100),
# ('NFLX', 100),
# ('NWL', 100),
# ('NFX', 100),
# ('NEM', 100),
# ('NWSA', 100),
# ('NWS', 100),
# ('NEE', 100),
# ('NLSN', 100),
# ('NKE', 100),
# ('NI', 100),
# ('NBL', 100),
# ('JWN', 100),
# ('NSC', 100),
# ('NTRS', 100),
# ('NOC', 100),
# ('NRG', 100),
# ('NUE', 100),
# ('NVDA', 100),
# ('ORLY', 100),
# ('OXY', 100),
# ('OMC', 100),
# ('OKE', 100),
# ('ORCL', 100),
# ('PCAR', 100),
# ('PH', 100),
# ('PDCO', 100),
# ('PAYX', 100),
# ('PYPL', 100),
# ('PNR', 100),
# ('PBCT', 100),
# ('PEP', 100),
# ('PKI', 100),
# ('PRGO', 100),
# ('PFE', 100),
# ('PCG', 100),
# ('PM', 100),
# ('PSX', 100),
# ('PNW', 100),
# ('PXD', 100),
# ('PNC', 100),
# ('RL', 100),
# ('PPG', 100),
# ('PPL', 100),
# ('PX', 100),
# ('PCLN', 100),
# ('PFG', 100),
# ('PG', 100),
# ('PGR', 100),
# ('PLD', 100),
# ('PRU', 100),
# ('PEG', 100),
# ('PSA', 100),
# ('PHM', 100),
# ('PVH', 100),
# ('QRVO', 100),
# ('QCOM', 100),
# ('PWR', 100),
# ('DGX', 100),
# ('RRC', 100),
# ('RTN', 100),
# ('O', 100),
# ('RHT', 100),
# ('REG', 100),
# ('REGN', 100),
# ('RF', 100),
# ('RSG', 100),
# ('RAI', 100),
# ('RHI', 100),
# ('ROK', 100),
# ('COL', 100),
# ('ROP', 100),
# ('ROST', 100),
# ('RCL', 100),
# ('R', 100),
# ('SPGI', 100),
# ('CRM', 100),
# ('SCG', 100),
# ('SLB', 100),
# ('SNI', 100),
# ('STX', 100),
# ('SEE', 100),
# ('SRE', 100),
# ('SHW', 100),
# ('SIG', 100),
# ('SPG', 100),
# ('SWKS', 100),
# ('SLG', 100),
# ('SNA', 100),
# ('SO', 100),
# ('LUV', 100),
# ('SWN', 100),
# ('SWK', 100),
# ('SPLS', 100),
# ('SBUX', 100),
# ('STT', 100),
# ('SRCL', 100),
# ('SYK', 100),
# ('STI', 100),
# ('SYMC', 100),
# ('SYF', 100),
# ('SYY', 100),
# ('TROW', 100),
# ('TGT', 100),
# ('TEL', 100),
# ('TGNA', 100),
# ('TDC', 100),
# ('TSO', 100),
# ('TXN', 100),
# ('TXT', 100),
# ('BK', 100),
# ('CLX', 100),
# ('COO', 100),
# ('HSY', 100),
# ('MOS', 100),
# ('TRV', 100),
# ('DIS', 100),
# ('TMO', 100),
# ('TIF', 100),
# ('TWX', 100),
# ('TJX', 100),
# ('TMK', 100),
# ('TSS', 100),
# ('TSCO', 100),
# ('TDG', 100),
# ('RIG', 100),
# ('TRIP', 100),
# ('FOXA', 100),
# ('FOX', 100),
# ('TSN', 100),
# ('USB', 100),
# ('UDR', 100),
# ('ULTA', 100),
# ('UA', 100),
# ('UAA', 100),
# ('UNP', 100),
# ('UAL', 100),
# ('UNH', 100),
# ('UPS', 100),
# ('URI', 100),
# ('UTX', 100),
# ('UHS', 100),
# ('UNM', 100),
# ('URBN', 100),
# ('VFC', 100),
# ('VLO', 100),
# ('VAR', 100),
# ('VTR', 100),
# ('VRSN', 100),
# ('VRSK', 100),
# ('VZ', 100),
# ('VRTX', 100),
# ('VIAB', 100),
# ('V', 100),
# ('VNO', 100),
# ('VMC', 100),
# ('WMT', 100),
# ('WBA', 100),
# ('WM', 100),
# ('WAT', 100),
# ('WEC', 100),
# ('WFC', 100),
# ('HCN', 100),
# ('WDC', 100),
# ('WU', 100),
# ('WRK', 100),
# ('WY', 100),
# ('WHR', 100),
# ('WFM', 100),
# ('WMB', 100),
# ('WLTW', 100),
# ('WYN', 100),
# ('WYNN', 100),
# ('XEL', 100),
# ('XRX', 100),
# ('XLNX', 100),
# ('XL', 100),
# ('XYL', 100),
# ('YHOO', 100),
# ('YUM', 100),
# ('ZBH', 100),
# ('ZION', 100),
# ('ZTS', 100)
