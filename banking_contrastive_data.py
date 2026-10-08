"""Deterministic banking-only minimal pairs in Nimble's retained record format."""
import hashlib, json, random
from pathlib import Path

TASKS = {
 "suspected_fraud": {
  "type":"noul","instructions":"Do the case facts indicate suspicious activity or possible fraud?",
  "criteria":{"false":"The facts affirm a recognized, expected event.","true":"The facts indicate an unrecognized or suspicious event."},
  "policy":"Treat an event as suspected fraud exactly when the customer reports that it is unrecognized.",
  "base":"The customer reports that the {channel} event involving {party} is recognized.",
  "counter":"The customer reports that the {channel} event involving {party} is unrecognized.","targets":[False,True]},
 "disputed_authorization": {
  "type":"noul","instructions":"Does the customer dispute authorizing the transaction?",
  "criteria":{"false":"The customer affirms authorization.","true":"The customer denies authorization."},
  "policy":"Authorization is disputed exactly when the customer denies authorizing the transaction.",
  "base":"The customer confirms they authorized the {channel} transaction to {party}.",
  "counter":"The customer denies they authorized the {channel} transaction to {party}.","targets":[False,True]},
 "conflicting_information": {
  "type":"noul","instructions":"Do material records in this case conflict with each other?",
  "criteria":{"false":"The material records agree.","true":"The material records disagree."},
  "policy":"Information conflicts exactly when the bank record and customer document state different material values.",
  "base":"The bank record and customer receipt both show {amount} for the {channel} payment.",
  "counter":"The bank record shows {amount}, but the customer receipt shows {other_amount} for the {channel} payment.","targets":[False,True]},
 "policy_exception": {
  "type":"noul","instructions":"Does resolving this request require an exception to a stated bank rule?",
  "criteria":{"false":"The request complies with the stated rule.","true":"The customer asks to bypass the stated rule."},
  "policy":"A policy exception is required exactly when the customer asks the bank to bypass the stated rule.",
  "base":"The customer accepts the stated {rule} and asks how to proceed within it.",
  "counter":"The customer rejects the stated {rule} and asks the bank to bypass it.","targets":[False,True]},
 "ongoing_loss": {
  "type":"noul","instructions":"Do the facts show unauthorized funds are currently leaving or about to leave?",
  "criteria":{"false":"No unauthorized transfer remains active or pending.","true":"An unauthorized transfer remains active or pending."},
  "policy":"Ongoing loss exists exactly when an unauthorized transfer remains active or pending.",
  "base":"The unauthorized {channel} transfer to {party} was stopped and no further transfer is pending.",
  "counter":"The unauthorized {channel} transfer to {party} was not stopped and a further transfer is pending.","targets":[False,True]},
 "customer_blocked": {
  "type":"noul","instructions":"Is the customer currently unable to access the account or complete the intended banking action?",
  "criteria":{"false":"The customer can access and complete the action.","true":"The customer cannot access or complete the action."},
  "policy":"The customer is blocked exactly when they cannot access the account or complete the intended action.",
  "base":"The customer can access the {account} and can complete the intended {action}.",
  "counter":"The customer cannot access the {account} and cannot complete the intended {action}.","targets":[False,True]},
 "department": {
  "type":"choice","instructions":"Which specialist team should own this case?",
  "criteria":{"payments":"Payment servicing for a transaction the customer recognizes.","fraud_review":"Fraud review for a transaction the customer denies authorizing."},
  "policy":"Route a recognized transaction-service request to payments; route a transaction the customer denies authorizing to fraud review.",
  "base":"The customer recognizes the {channel} transaction to {party} and asks for its status.",
  "counter":"The customer denies authorizing the {channel} transaction to {party} and asks for investigation.","targets":["payments","fraud_review"]},
 "deadline_pressure": {
  "type":"score","instructions":"Ignoring fraud, account blockage, and ongoing loss, rate the independent deadline pressure.",
  "criteria":["Routine: no deadline within the next week.","Prompt: a stated deadline is within one business day.","Immediate: a stated deadline is within hours."],
  "policy":"Use Routine for no deadline within a week, Prompt for a deadline within one business day, and Immediate for a deadline within hours.",
  "base":"The request has no deadline within the next week.",
  "counter":"The request has a deadline within two hours.","targets":[0,2]},
}

VALUES={"channel":["card","wire","ACH","bill-pay","mobile-transfer","cash-withdrawal"],"party":["a utility","a supplier","a landlord","a merchant","a relative","a contractor"],
        "amount":["$18","$74","$240","$890","$1,600","$4,200"],"other_amount":["$19","$79","$204","$980","$1,060","$4,020"],
        "rule":["daily transfer limit","identity-verification rule","beneficiary hold","cooling-off period","recovery requirement","approval limit"],
        "account":["checking account","savings account","business account","online profile","joint account","mobile profile"],
        "action":["payment","wire","statement download","beneficiary update","card lock","profile update"]}

def make(split,pairs,seed):
    rng=random.Random(seed);rows=[];names=list(TASKS)
    for i in range(pairs):
        name=names[i%len(names)];task=TASKS[name];vals={k:rng.choice(v) for k,v in VALUES.items()};family=f"bank-{split}-{i:04d}"
        question={"type":task["type"],"instructions":task["instructions"],"criteria":task["criteria"]}
        for variant,index in (("base",0),("counterfactual",1)):
            fact=task["base" if index==0 else "counter"].format(**vals);target=task["targets"][index]
            state={"policy":task["policy"],"case_fact":fact,"request":"Apply the policy to the case fact."}
            certificate={"necessity_checks_passed":True,"focus_fact":name,"base_and_counterfactual_differ":True,
                         "remove_policy":"unknown","remove_case_fact":"unknown","no_answer_leakage":True}
            rows.append({"id":f"{family}-{variant}","family":family,"source_family":family,"domain":"banking","split":split,
                         "variant":variant,"input":{"state":state,"questions":{"decision":question}},
                         "reference":{"target":target,"source":"deterministic_policy_over_minimal_pair","human_reviewed":False},
                         "evidence_certificate":certificate})
    return rows

def build():return {"train":make("train",256,7101),"validation":make("validation",64,8202),"test":make("test",64,9303)}

def save(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);data=build();manifest={}
    families={s:{r['source_family'] for r in rows} for s,rows in data.items()}
    assert all(not a&b for i,a in enumerate(families.values()) for b in list(families.values())[i+1:])
    for split,rows in data.items():
        body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);path=directory/(split+'.jsonl')
        if path.exists() and path.read_text()!=body:raise ValueError('Frozen dataset differs; use a new directory')
        path.write_text(body);manifest[split]={"rows":len(rows),"pairs":len(rows)//2,"sha256":hashlib.sha256(body.encode()).hexdigest(),"tasks":dict(__import__('collections').Counter(r['evidence_certificate']['focus_fact'] for r in rows))}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return data,manifest
