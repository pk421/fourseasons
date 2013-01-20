#!/bin/sh

kill_python:
	killall -9 python

start_services:
	#./start_services.sh
	supervisord --configuration etc/supervisord.conf

stop_services:
	supervisorctl --configuration etc/supervisord.conf stop all
	supervisorctl --configuration etc/supervisord.conf shutdown

poll_start:
	supervisorctl --configuration etc/supervisord.conf start poll_realtime_data

poll_stop:
	supervisorctl --configuration etc/supervisord.conf stop poll_realtime_data

redis_start:
	supervisorctl --configuration etc/supervisord.conf start redis

redis_stop:
	supervisorctl --configuration etc/supervisord.conf stop redis

download_stocks:
	python test_run.py --download_stocks

extract_symbols:
	python test_run.py --extract_symbols_with_historical_data

load_redis:
	python test_run.py --load_redis

poll_realtime_data:
	python test_run.py --poll_realtime_data

read_redis:
	python test_run.py --read_redis

stock_analyzer:
	python test_run.py --stock_analyzer

vol_analyzer:
	python test_run.py --vol_analyzer
