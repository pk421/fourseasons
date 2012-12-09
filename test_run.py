import argparse
import os
import time

from src.data_retriever import *
from src.poll_realtime_data import *
from src.stock_analyzer import run_stock_analyzer

parser = argparse.ArgumentParser()
#parser.add_argument("square",
#                    help="display a square of a given number",
#                    type=int)
#parser.add_argument("-v", "--verbosity",
#                    help="increase output verbosity",
#                    action="store_true")
parser.add_argument("--download_stocks",
                    help="run multithreaded download from yahoo",
                    action="store_true")

parser.add_argument("--extract_symbols_with_historical_data",
                    help="find symbols that have historical data",
                    action="store_true")

parser.add_argument("--load_redis",
                    help="move stock historical data in objects, and then into redis validating data at the same time",
                    action="store_true")

parser.add_argument("--poll_realtime_data",
                    help="test and engage a screenscraper to continuously feed in a data stream of prices and times",
                    action="store_true")

parser.add_argument("--read_redis",
                    help="read data from redis and put it in python objects in memory",
                    action="store_true")

parser.add_argument("--stock_analyzer",
                    help="run the analysis backend for some securities",
                    action="store_true")

args = parser.parse_args()
#print args.square**2
#if args.verbosity:
#    print "verbosity turned on"

time_start = time.time()

if args.download_stocks:
    multithread_yahoo_download('300B_1M.csv', thread_count=5, update_check=True, new_only=False)

if args.extract_symbols_with_historical_data:
    extract_symbols_with_historical_data()

if args.load_redis:
    load_redis(stock_list='list_sp_500.csv', db_number=14, file_location='data/test/')

if args.poll_realtime_data:
    query_realtime_data()

if args.read_redis:
    read_redis(db_number = 1, to_disk=True)

if args.stock_analyzer:
    run_stock_analyzer()



time_end = time.time()
time_total = round(time_end - time_start, 4)
print "\n\nTime Needed: ", time_total, " sec"