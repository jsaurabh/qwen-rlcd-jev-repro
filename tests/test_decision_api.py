import copy
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from fastapi.testclient import TestClient
from decision_api import DecisionsClient, SystemOneClient, APIError
from decision_api.protocol import answer, decisions_to_systemone, systemone_to_decisions
from decision_api.server import create_app
from examples.banking_decisions import STATE, QUESTIONS


class Backend:
    def __init__(self): self.rows=[]
    def predict(self, rows):
        self.rows=rows
        return [[.1,.9] if r['question']['type']=='noul' else [.1,.2,.7] for r in rows],120


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.backend=Backend();self.client=TestClient(create_app(self.backend,api_key='test-key'))
        self.body={'model':'our-autojev','input':STATE,'questions':copy.deepcopy(QUESTIONS)}
        self.headers={'Authorization':'Bearer test-key'}
    def post(self,body=None):return self.client.post('/v1/decisions',json=body or self.body,headers=self.headers)
    def test_both_routes_identical_distributions(self):
        d=self.post();self.assertEqual(d.status_code,200)
        body,mapping=decisions_to_systemone(self.body)
        s=self.client.post('/v1/systemone',json=body,headers=self.headers)
        self.assertEqual(d.json(),systemone_to_decisions(s.json(),mapping))
        answers=d.json()['answers']
        self.assertEqual(answers[0]['probability'],.9)
        self.assertEqual(answers[1]['choice'],'fraud_review')
        self.assertAlmostEqual(answers[2]['score'],1.6)
        self.assertEqual([a['name'] for a in answers],['review','team','urgency'])
        self.assertEqual(d.json()['usage']['input_tokens'],120)
    def test_typed_values_do_not_collide(self):
        body,mapping=decisions_to_systemone({'model':'x','input':'x','questions':[{'type':'choice','instructions':'x','choices':[{'value':True},{'value':'true'},{'value':False}]}]})
        q=body['questions']['q0']
        response={'model':'x','answers':{'q0':answer(q,[.8,.1,.1])}}
        result=systemone_to_decisions(response,mapping)['answers'][0]
        self.assertIs(result['choice'],True)
        self.assertEqual([p['value'] for p in result['probabilities']],[True,'true',False])
    def test_invalid_requests(self):
        for changes in [{'input':[{'role':'user','content':'hello'}]}, {'extra':1}, {'safety_identifier':'abc'}, {'questions':[]}, {'model':'different'}]:
            body={**self.body,**changes}
            self.assertIn(self.post(body).status_code,(422,404))
        body=copy.deepcopy(self.body);body['questions'][1]['name']='review'
        self.assertEqual(self.post(body).status_code,422)
        body=copy.deepcopy(self.body);body['questions'][1]['choices'][1]['value']='payments'
        self.assertEqual(self.post(body).status_code,422)
    def test_auth_and_size(self):
        self.assertEqual(self.client.post('/v1/decisions',json=self.body).status_code,401)
        self.assertEqual(self.client.post('/v1/decisions',content=b'x'*(1024*1024+1),headers=self.headers).status_code,413)
    def test_backend_invalid_probability_is_500(self):
        self.backend.predict=lambda rows: ([[float('nan')]]*len(rows),0)
        self.assertEqual(self.post().status_code,500)
    def test_score_is_expected_index_not_argmax(self):
        q={'type':'score','criteria':['low','medium','high']}
        self.assertAlmostEqual(answer(q,[.1,.7,.2])['score'],1.1)
        self.assertEqual(answer(q,[1/3]*3)['confidence'],0)
    def test_noul_no_confidence(self):
        self.assertEqual(answer({'type':'noul'},[.5,.5]),{'type':'noul','noul':.5})
    def test_structured_state_and_noul_criteria(self):
        body={'model':'our-autojev','state':{'message':'hello'},'questions':{'q':{'type':'noul','instructions':{'question':'Fraud?'},'criteria':{'true':'Unauthorized','false':'Authorized'}}}}
        self.assertEqual(self.client.post('/v1/systemone',json=body,headers=self.headers).status_code,200)
        self.assertEqual(self.backend.rows[0]['state'],body['state'])
    def test_refusal_preserved(self):
        _,mapping=decisions_to_systemone(self.body)
        result=systemone_to_decisions({'model':'x','answers':{m['key']:{'type':'refusal'} for m in mapping}},mapping)
        self.assertTrue(all(a['type']=='refusal' and 'probabilities' not in a for a in result['answers']))


class HTTPClientTests(unittest.TestCase):
    def test_actual_http_client_and_error(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.path,body,self.headers.get('Authorization')))
                status=401 if body['model']=='bad' else 200
                self.send_response(status);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(json.dumps({'model':body['model'],'answers':[],'usage':{}}).encode())
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}/v1/'
            c=DecisionsClient(base_url=url,api_key='test-key')
            c.create(model='our-autojev',input=STATE,questions=QUESTIONS)
            self.assertEqual(requests[-1][0],'/v1/decisions')
            self.assertEqual(requests[-1][2],'Bearer test-key')
            SystemOneClient(base_url=url).create(model='x',state='x',questions={})
            self.assertEqual(requests[-1][0],'/v1/systemone')
            with self.assertRaises(APIError) as error:c.create(model='bad',input='x',questions=[])
            self.assertEqual(error.exception.status,401)
            self.assertNotIn('test-key',str(error.exception))
        finally:server.shutdown();server.server_close();thread.join()
    def test_url_validation(self):
        for url in ['http://external.example/v1','https://user:pass@example.org/v1','https://example.org/v1?key=foo']:
            with self.assertRaises(ValueError):DecisionsClient(base_url=url)


if __name__=='__main__':unittest.main()
