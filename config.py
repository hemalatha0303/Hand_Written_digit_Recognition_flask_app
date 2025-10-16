from os import environ as env
import multiprocessing


HOST = env.get('HOST', '127.0.0.1')
PORT = int(env.get('PORT', 5000))
DEBUG_MODE = bool(int(env.get('DEBUG_MODE', '0')))


# Gunicorn config (if you run via gunicorn)
bind = ':' + str(PORT)
workers = max(2, multiprocessing.cpu_count() * 2 + 1)
threads = multiprocessing.cpu_count() * 2