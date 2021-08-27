# import pyximport
# pyximport.install()

import argparse
import os
import time

try:
    from src.cointegrations import run_cointegrations
    from src.correlations import run_correlations
    from src.data_retriever import *
    from src.poll_realtime_data import *
    from src.portfolio_analysis.portfolio_analysis import run_portfolio_analysis
    from src.portfolio_analysis.portfolio_analysis import run_live_portfolio_analysis
    from src.portfolio_analysis.sweep_portfolios import run_sweep_portfolios
    from src.harding_seasonality import run_harding_seasonality
    from src.indicator_system import run_indicator_system
    from src.returns_analyzer import run_returns_analyzer
    from src.stock_analyzer import run_stock_analyzer
    from src.vol_analyzer import run_vol_analyzer
    
    from src.live_monitor.live_monitor import run_live_monitor

except:
    print "\n\nCould not make all imports"

parser = argparse.ArgumentParser()
#parser.add_argument("square",
#                    help="display a square of a given number",
#                    type=int)
#parser.add_argument("-v", "--verbosity",
#                    help="increase output verbosity",
#                    action="store_true")
parser.add_argument("--cointegrations",
                    help="run cointegration analysis",
                    action="store_true")

parser.add_argument("--correlations",
                    help="run correlation analysis",
                    action="store_true")

parser.add_argument("--download_stocks",
                    help="run multithreaded download from yahoo",
                    action="store_true")

parser.add_argument("--extract_symbols_with_historical_data",
                    help="find symbols that have historical data",
                    action="store_true")

parser.add_argument("--harding_seasonality",
                    help="Analyze a seasonal system with simple technical indicators.",
                    action="store_true")

parser.add_argument("--indicator_system",
                    help="Simulate a trade system similar to --cointegrations, but use more normal indicators.",
                    action="store_true")

parser.add_argument("--live_monitor",
                    help="Run the live monitor and email system.",
                    action="store_true")

parser.add_argument("--live_portfolio_analysis",
                    help="run live portfolio analysis",
                    action="store_true")

parser.add_argument("--load_redis",
                    help="move stock historical data in objects, and then into redis validating data at the same time",
                    action="store_true")

parser.add_argument("--poll_realtime_data",
                    help="test and engage a screenscraper to continuously feed in a data stream of prices and times",
                    action="store_true")

parser.add_argument("--portfolio_analysis",
                    help="run a simulation with weightings in a portfolio",
                    action="store_true")

parser.add_argument("--read_redis",
                    help="read data from redis and put it in python objects in memory",
                    action="store_true")

parser.add_argument("--returns_analyzer",
                    help="run the returns analyzer for some securities",
                    action="store_true")

parser.add_argument("--stock_analyzer",
                    help="run the analysis backend for some securities",
                    action="store_true")

parser.add_argument("--sweep_portfolios",
                    help="Similar to portfolio_analysis, but allows sweeping over many asset groupings.",
                    action="store_true")

parser.add_argument("--vol_analyzer",
                    help="run the volatility analysis backend for some securities",
                    action="store_true")


args = parser.parse_args()
#print args.square**2
#if args.verbosity:
#    print "verbosity turned on"

time_start = time.time()

if args.cointegrations:
    run_cointegrations()

if args.correlations:
    run_correlations()

if args.download_stocks:
    """
    if update_check = True, then this will will NOT overwrite existing files in folder
    if new_only = True, then this will NOT download any files that already exist in the folder
    """
    # multithread_yahoo_download('list_sp_500.csv', thread_count=6, update_check=False, new_only=False, \
    #                             store_location = 'tmp/')
    # multithread_yahoo_download('300B_1M_and_etfs_etns.csv', thread_count=6, update_check=False, new_only=False, \
    #                             store_location = 'tmp/')
    multithread_yahoo_download('test.csv', thread_count=1, update_check=False, new_only=False, \
                                store_location = 'tmp/')
    # multithread_yahoo_download('300B_1M_and_etfs_etns.csv', thread_count=6, update_check=False, new_only=True)

if args.extract_symbols_with_historical_data:
    extract_symbols_with_historical_data()

if args.harding_seasonality:
    run_harding_seasonality()

if args.indicator_system:
    run_indicator_system()

if args.live_monitor:
    run_live_monitor()

if args.live_portfolio_analysis:
    run_live_portfolio_analysis()

if args.load_redis:
    # load_redis(stock_list='list_sp_500.csv', db_number=0, file_location='tmp/', dict_size=3)
    # load_redis(stock_list='300B_1M_and_etfs_etns.csv', db_number=0, file_location='tmp/', dict_size=3)
    load_redis(stock_list='test.csv', db_number=0, file_location='tmp/', dict_size=3)
    # load_redis(stock_list='other_data.csv', db_number=1, file_location='tmp/', dict_size=3)

if args.poll_realtime_data:
    query_realtime_data()

if args.portfolio_analysis:
    run_portfolio_analysis()

if args.read_redis:
    read_redis(db_number=1, to_disk=True)

if args.returns_analyzer:
    run_returns_analyzer()

if args.stock_analyzer:
    run_stock_analyzer()

"""
This module is similar to portfolio_analysis, but it is aimed at sweeping over many different groupings of assets to
find which group has the best DR at a given point in time. The eventual goal would be a full simulation, where assets
can effectively be swapped in and out.
"""
if args.sweep_portfolios:
    run_sweep_portfolios()

if args.vol_analyzer:
    for x in xrange(0, 1):
        run_vol_analyzer()
        print x


time_end = time.time()
time_total = round(time_end - time_start, 4)
print "\n\nFinish Time, Time Needed: ", time.asctime(), '\t', time_total, " sec", '\n'
