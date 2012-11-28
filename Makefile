#!/bin/sh

kill_python:
	killall -9 python

start_services:
	#./start_services.sh
	supervisord --configuration etc/supervisord.conf

stop_services:
	supervisorctl --configuration etc/supervisord.conf stop all
	supervisorctl --configuration etc/supervisord.conf shutdown

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

