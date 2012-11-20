#!/bin/sh

kill_python:
	killall -9 python

start_services:
	$(MAKE) kill_python
	./start_services.sh

download_stocks:
	$(MAKE) kill_python
	python test_run.py --download_stocks

extract_symbols:
	$(MAKE) kill_python
	python test_run.py --extract_symbols_with_historical_data

load_redis:
	$(MAKE) kill_python
	python test_run.py --load_redis

poll_realtime_data:
	$(MAKE) kill_python
	python test_run.py --poll_realtime_data

read_redis:
	$(MAKE) kill_python
	python test_run.py --read_redis