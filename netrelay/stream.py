import os
import sys
import json
import pycurl
from multiprocessing import shared_memory


MAX_BYTE = 32768
shm_result = shared_memory.ShareableList([' ' * MAX_BYTE], name='curl_result')

def callback(data):
    shm_result[0] = data


# Start server, needs to execute PST Rest Server first.
OPEN_TRACKER_COMMAND = 'curl --header "Content-Type: application/json" --request POST --data "\{\}" http://localhost:7278/PSTapi/Start'
os.system(OPEN_TRACKER_COMMAND)


c = pycurl.Curl()
c.setopt(pycurl.URL, 'http://localhost:7278/PSTapi/StartTrackerDataStream')
c.setopt(pycurl.WRITEFUNCTION, callback)
try:
    c.perform()
except pycurl.error:
    pass

c.close()
