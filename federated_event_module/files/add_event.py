import numpy
import math
from pymisp import ExpandedPyMISP, PyMISP, MISPEvent
import coloredlogs, logging
import os
import time

coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(name)s[%(process)d] %(levelname)s %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

time.sleep(60)

misp_url = 'https://localhost:6666'
misp_key = "5cb1b97269eeb2b5143a45ff3df41cf49306ffd5"
misp_verifycert = False

misp = ExpandedPyMISP(misp_url, misp_key, misp_verifycert)


event_obj = MISPEvent()
event_obj.distribution = 1
event_obj.threat_level_id = 2
event_obj.analysis = 1
event_obj.info = "New threat alert"

event = misp.add_event(event_obj)

event_id, event_uuid = event['Event']['id'], event['Event']['uuid']
logging.info("Event has been added to the local MISP instance")
