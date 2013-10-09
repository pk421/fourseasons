import datetime
import time

import redis



def run_live_monitor():

	send_email()

	return


def send_email():

	import smtplib

	from_address = 'mikesserver@dontreply.com'
	to_address = 'mcpilat@gmail.com'

	msg = '\n'
	msg += 'test email in python and gmail'

	username = 'mcpilat'
	password = 'montana316'

	server = smtplib.SMTP('smtp.gmail.com:587')
	server.starttls()
	server.login(username,password)
	server.sendmail(from_address, to_address, msg)
	server.quit()

	print "Finished sending"