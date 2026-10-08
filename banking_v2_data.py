"""Deterministic synthetic banking cases for the compositional v2 adapter."""
import hashlib
import json
import random
from pathlib import Path

FIELDS = [
    {"name":"department","type":"enum","question":"Which specialist team should own this case? Choose payments for payment servicing, account_support for account access or profile servicing, fraud_review for suspicious or disputed activity, and other for product inquiries.","choices":["payments","account_support","fraud_review","other"]},
    {"name":"suspected_fraud","type":"noul","question":"Do the observed facts indicate suspicious activity or possible fraud?"},
    {"name":"disputed_authorization","type":"noul","question":"Does the customer dispute authorizing a transaction or account action?"},
    {"name":"conflicting_information","type":"noul","question":"Do material records or identity facts conflict with each other?"},
    {"name":"policy_exception","type":"noul","question":"Does resolving the request require an exception to a stated bank policy or limit?"},
    {"name":"ongoing_loss","type":"noul","question":"Do the observed facts show money is currently leaving or is about to leave without authorization?"},
    {"name":"customer_blocked","type":"noul","question":"Is the customer currently unable to access the account or complete the intended banking action?"},
    {"name":"deadline_pressure","type":"enum","question":"Ignoring fraud, ongoing loss, and account blockage, what independent time pressure is stated? Choose routine for no deadline, prompt for a near deadline, or immediate for a deadline within hours.","choices":["routine","prompt","immediate"]},
]

ARCHETYPES = [
 ("recognized_payment","payments",0,0,0,0,0,0,"routine"),
 ("payment_blocked","payments",0,0,0,0,0,1,"prompt"),
 ("payment_conflict","payments",0,0,1,0,0,1,"prompt"),
 ("payment_exception","payments",0,0,0,1,0,1,"routine"),
 ("routine_account","account_support",0,0,0,0,0,0,"routine"),
 ("login_blocked","account_support",0,0,0,0,0,1,"routine"),
 ("identity_conflict","account_support",0,0,1,0,0,1,"prompt"),
 ("recovery_exception","account_support",0,0,0,1,0,1,"routine"),
 ("historic_fraud","fraud_review",1,1,0,0,0,0,"prompt"),
 ("ongoing_fraud","fraud_review",1,1,0,0,1,0,"routine"),
 ("suspicious_access","fraud_review",1,0,0,0,0,1,"routine"),
 ("product_inquiry","other",0,0,0,0,0,0,"routine"),
]

TEMPLATES = {
 "recognized_payment":"The customer recognizes the {channel} payment of {amount} to {party} and only wants {detail}; it is within the expected timeline.",
 "payment_blocked":"An authorized {channel} payment of {amount} to {party} has failed, preventing {goal}; the customer reports no suspicious activity and needs action before {deadline}.",
 "payment_conflict":"The customer authorized a {channel} payment to {party}, but the bank record and receipt show different {detail}, so {goal} cannot proceed before {deadline}.",
 "payment_exception":"An authorized {channel} payment to {party} is blocked by the stated {policy}; the customer asks the bank to waive that rule so they can complete {goal}.",
 "routine_account":"The signed-in customer wants {detail} for the {account}; access works normally and no transaction is disputed.",
 "login_blocked":"The customer cannot access the {account} because {access_issue}; they report no suspicious activity or disputed transaction.",
 "identity_conflict":"Recovery for the {account} is blocked because {identity_a} conflicts with {identity_b}; access is needed before {deadline}.",
 "recovery_exception":"The customer cannot access the {account} and asks the bank to bypass the stated {policy}; no suspicious activity is reported.",
 "historic_fraud":"The customer denies authorizing a completed {channel} transaction of {amount} to {party}; credentials are secured and no further loss is pending, but review is requested before {deadline}.",
 "ongoing_fraud":"The customer denies authorizing a {channel} transaction of {amount} to {party}; it is currently pending or repeating and funds may leave within hours. Access has not stopped it.",
 "suspicious_access":"An unfamiliar device accessed the {account}; the bank has not confirmed a transaction, and the customer is locked out while the alert is reviewed.",
 "product_inquiry":"The customer asks for general information about {product}, with no access problem, transaction dispute, policy exception, or deadline.",
}

VALUES = {
 "channel":["card","wire","ACH","bill-pay","cash withdrawal","mobile transfer"],
 "amount":["$18","$74","$240","$890","$1,600","$4,200"],
 "party":["a utility","a supplier","a landlord","a merchant","a relative","a contractor"],
 "detail":["a receipt","the settlement date","the statement label","the reference number","the posted amount","notification settings"],
 "goal":["a payroll run","a purchase","a rent payment","a shipment","a service renewal","a closing"],
 "deadline":["tomorrow morning","the end of day","the next business day","a scheduled closing","the billing cutoff","the payroll cutoff"],
 "policy":["daily transfer limit","identity verification rule","cooling-off period","beneficiary hold","recovery requirement","approval limit"],
 "account":["checking account","savings account","business account","online profile","joint account","mobile banking profile"],
 "access_issue":["the verification code never arrives","the password reset fails","the login is locked","the authentication app rejects every code","the recovery link has expired","the profile page will not load"],
 "identity_a":["the submitted legal name","the recovery phone number","the ownership document","the date of birth","the business registration","the mailing address"],
 "identity_b":["the bank record","the account application","the verified profile","the tax record","the signature card","the previous verification"],
 "product":["mortgage rates","savings products","student accounts","business credit","safe-deposit boxes","foreign currency services"],
}

def _target(index, size, certainty=0.96):
    rest=(1-certainty)/(size-1)
    out=[rest]*size;out[index]=certainty
    return out

def generate(split, count, seed):
    rng=random.Random(seed); rows=[]; seen=set()
    dept=["payments","account_support","fraud_review","other"]
    deadline=["routine","prompt","immediate"]
    for i in range(count):
        a=ARCHETYPES[i%len(ARCHETYPES)]
        name,d,sf,da,ci,pe,ol,cb,dp=a
        vals={k:rng.choice(v) for k,v in VALUES.items()}
        message=TEMPLATES[name].format(**vals)
        # Additional observed fact combinations make each case a distinct state, not a wrapper copy.
        case={"case_reference":f"SYN-{rng.randrange(100000,999999)}","customer_message":message,
              "observed_facts":{"account_secured":name=="historic_fraud","transaction_pending":name=="ongoing_fraud",
                                "customer_access_available":not bool(cb)},
              "policy_context":{"deadline":"within hours" if dp=="immediate" else vals["deadline"] if dp=="prompt" else "none stated",
                                "exception_requested":bool(pe),"named_rule":vals["policy"] if pe else "none"}}
        state=json.dumps(case,separators=(",",":"),sort_keys=True)
        assert state not in seen;seen.add(state)
        labels=[dept.index(d),sf,da,ci,pe,ol,cb,deadline.index(dp)]
        certainty=.76 if name=="suspicious_access" else .96
        targets=[]
        for j,y in enumerate(labels):
            c=certainty if j in (0,1) else .96
            targets.append(_target(y,len(FIELDS[j].get("choices",[False,True])),c))
        rows.append({"id":f"{split}-{i:04d}","group":name,"archetype":name,"state":state,
                     "labels":labels,"targets":targets})
    return rows

def build():
    data={"train":generate("train",240,3101),"validation":generate("validation",72,4202),"test":generate("test",120,5303)}
    states=[{r["state"] for r in data[s]} for s in data]
    assert all(not a&b for i,a in enumerate(states) for b in states[i+1:])
    return data

def save(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);data=build();manifest={}
    for split,rows in data.items():
        body="".join(json.dumps(r)+"\n" for r in rows);path=directory/(split+".jsonl")
        if path.exists() and path.read_text()!=body: raise ValueError("Frozen dataset differs; use a new output directory")
        path.write_text(body);manifest[split]={"sha256":hashlib.sha256(body.encode()).hexdigest(),"cases":len(rows),"archetypes":len({r["archetype"] for r in rows})}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return data,manifest
