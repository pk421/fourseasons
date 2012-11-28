from src.data_retriever import *
from src.poll_realtime_data import *
import argparse
import os

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

args = parser.parse_args()
#print args.square**2
#if args.verbosity:
#    print "verbosity turned on"

if args.download_stocks:
    multithread_yahoo_download('300B_1M.csv', thread_count=1, update_check=True, new_only=False)

if args.extract_symbols_with_historical_data:
    extract_symbols_with_historical_data()

if args.load_redis:
    load_redis(stock_list='300B_1M.csv')

if args.poll_realtime_data:
    query_realtime_data()

if args.read_redis:
    read_redis()
