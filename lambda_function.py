import requests
import json
import os
import re

url = os.environ.get('MODEL_URL')+'/v2/models/ensemble-drug-interaction/infer' #API URL


def lambda_handler(event, context):
	msg = {}
	try:
		input_dict = json.loads(json.dumps(event))
		drug1 = input_dict['smile1']
		drug2 = input_dict['smile2']
		json_req = {
					  "inputs" : [
						{
							"name" : "INPUT1",
							"shape" : [ 1, 1],
							"datatype" : "BYTES",
							"data" : [drug1]
						},
						{
							"name" : "INPUT2",
							"shape" : [ 1, 1],
							"datatype" : "BYTES",
							"data" : [drug2]
						}
						]
					}
		
		x = requests.post(url, json=json_req)
		data = json.loads(x.text)
		sample=data['outputs'][0]['data']
		p = re.compile('(?<!\\\\)\'')
		lst = []
		for i, _ in enumerate(sample):
			lst.append(json.loads(p.sub('\"', sample[i])))
		#return lst#json.loads(json.dumps(data['outputs'][0]['data']))
		msg.update({"code": x.status_code})
		if x.status_code == 200:
			msg.update({"result": lst})
			return json.loads(json.dumps(msg, indent=4))
		else:
			msg.update({"result": ['Error']})
			raise Exception('Internal Error: Error getting inference from model')  
	except Exception as e:
		print(e)
		msg.update({"code":  0})
		msg.update({"result":  str(e)})
		raise Exception('Internal Error: Error getting inference from model')