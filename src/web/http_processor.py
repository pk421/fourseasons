import datetime
import json

import tornado.escape
import tornado.ioloop
import tornado.web

import redis

class VersionHandler(tornado.web.RequestHandler):
	def get(self):
		response = { 'version': '3.5.1',
					 'last_build': datetime.date.today().isoformat() }
		self.write(response)

class GetGameByIdHandler(tornado.web.RequestHandler):
	def get(self, id):
		response = { 'id' : int(id),
					 'name' : 'Crazy Game',
					 'release_date' : datetimedate.today().isoformat() }
		self.write(response)

class CurrentResultsHandler(tornado.web.RequestHandler):
	def get(self):

		redis_reader = redis.StrictRedis(host='localhost', port=6379, db=14)
		current_result = redis_reader.get('json_result')
		
		current_result = json.loads(current_result)
		current_time = str(datetime.datetime.now())
		current_result.insert(0, current_time)
		current_result = json.dumps(current_result)

		# final_result = convert_to_json(input=current_result)
		final_result = current_result

		self.write(str(final_result))

		
application = tornado.web.Application([
	(r"/getgamebyid/([0-9]+)", GetGameByIdHandler),
	(r"/version", VersionHandler),
	(r"/current_results", CurrentResultsHandler)
])


if __name__ == "__main__":
	application.listen(8888)
	tornado.ioloop.IOLoop.instance().start()