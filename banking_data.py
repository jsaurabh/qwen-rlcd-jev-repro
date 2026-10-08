"""Fixed illustrative banking rubric data. No real customers or institutional policy."""
import hashlib
import json
from pathlib import Path

# Each row is an independently authored situation with department/urgency/review labels.
# Indices: payments/account_support/fraud_review; urgency 0/1/2; review false/true.
TRAIN = [
 ('Please explain a merchant charge that I recognize; I am not disputing it.', [0,0,0]),
 ('What is the expected arrival date of my scheduled wire? It is not late.', [0,0,0]),
 ('I need a receipt for an authorized bill payment. Everything is working.', [0,0,0]),
 ('How do I schedule an ordinary recurring payment? There is no issue with access.', [0,0,0]),
 ('My authorized transfer failed, so I cannot complete a purchase. No fraud is suspected.', [0,1,0]),
 ('An expected payment is overdue and the recipient cannot proceed. No authorization is disputed.', [0,1,0]),
 ('I need an exception to a payment limit to unblock a transfer. No funds are being stolen.', [0,1,1]),
 ('I recognize the merchant but the payment amounts in two records conflict; settlement is blocked.', [0,1,1]),
 ('How can I update my contact address? My account works normally.', [1,0,0]),
 ('Where can I download my account statements? I can sign in.', [1,0,0]),
 ('I would like to change my account nickname. Nothing is broken.', [1,0,0]),
 ('Which screen shows my account details? I am already signed in.', [1,0,0]),
 ('I forgot my password and cannot access my account. No suspicious activity is reported.', [1,1,0]),
 ('The login verification code is not arriving and I cannot sign in. No fraud is suspected.', [1,1,0]),
 ('Account access cannot be restored because the identity records contradict each other.', [1,1,1]),
 ('Please bypass the account recovery policy to restore my access. No activity is unauthorized.', [1,1,1]),
 ('I dispute a past card payment. The card is blocked and there is no ongoing loss.', [2,1,1]),
 ('A completed transfer was not authorized by me. The account is now secured.', [2,1,1]),
 ('An attacker is transferring money out of my account right now.', [2,2,1]),
 ('Another withdrawal by a thief is pending and has not been stopped.', [2,2,1]),
 ('Stolen credentials are being used to drain my balance now.', [2,2,1]),
 ('An unauthorized card transaction is in progress and my card remains active.', [2,2,1]),
 ('I received a suspicious login alert. Access is blocked and the bank confirmed there are no pending transactions.', [2,1,1]),
 ('A stranger has just logged in and initiated a transfer which is still pending.', [2,2,1]),
]
VALIDATION = [
 ('The customer wants to know how an already-recognized service payment appears on the statement; no dispute or disruption.', [0,0,0]),
 ('The customer requests the reference number for a successfully completed authorized wire.', [0,0,0]),
 ('A consented payment was rejected and is blocking the customer from placing an order; no fraud concerns.', [0,1,0]),
 ('Processing of a known payment is stuck because two confirmation records disagree on its amount.', [0,1,1]),
 ('The customer asks where account notification preferences can be edited; login works.', [1,0,0]),
 ('The customer requests instructions to view their account number while already logged in.', [1,0,0]),
 ('A broken login screen prevents normal access; the customer sees no signs of unauthorized activity.', [1,1,0]),
 ('Account recovery is stalled because the customer requests an exception to identity-verification policy.', [1,1,1]),
 ('A historic withdrawal is disputed as unauthorized; all access is secured and no more withdrawals are pending.', [2,1,1]),
 ('The customer denies authorizing a settled card purchase. The card is locked and further loss is ruled out.', [2,1,1]),
 ('A criminal currently controls online banking and is sending additional funds out.', [2,2,1]),
 ('The customer sees an unrecognized outgoing wire due to execute imminently; it has not been canceled.', [2,2,1]),
]
TEST = [
 ('I approved my electricity payment. Can you tell me when it will arrive? It is still within the expected timeframe.', [0,0,0]),
 ('Please explain the description of a purchase I recognize. I am asking for information, not disputing it.', [0,0,0]),
 ('I want to learn how to set a future payment date. My account and payments are functioning.', [0,0,0]),
 ('Can I obtain proof that yesterday\'s authorized payment completed? There is no payment problem.', [0,0,0]),
 ('An authorized merchant payment is repeatedly failing, preventing my order. There is no unauthorized activity.', [0,1,0]),
 ('The recipient cannot complete their transaction because my authorized remittance is stuck past its delivery date.', [0,1,0]),
 ('My approved bill payment remains unprocessed and is disrupting service. No policy exception or fraud is involved.', [0,1,0]),
 ('I want special permission to exceed the stated remittance limit because my payment is blocked.', [0,1,1]),
 ('Both sides agree a payment was authorized, but their receipts show different amounts and processing is suspended.', [0,1,1]),
 ('An authorized transfer is being held because the customer asks for a waiver of a stated payment-policy requirement.', [0,1,1]),
 ('I am signed in and would like instructions for switching to electronic account statements.', [1,0,0]),
 ('Where do I update the telephone number on my account? I have normal access and no urgent issue.', [1,0,0]),
 ('Please show me where to find my account-opening date. My account is operating normally.', [1,0,0]),
 ('I am looking for the menu to rename a savings account. I do not have an access or transaction problem.', [1,0,0]),
 ('I mistyped my password too often and now cannot log in. I have seen no suspicious activity.', [1,1,0]),
 ('The authentication app is not producing a working code, so I cannot access online banking. No fraud is suspected.', [1,1,0]),
 ('My profile page will not load and prevents me from accessing the account. There is no suspicious activity.', [1,1,0]),
 ('Access recovery has been halted: the two ownership documents identify different people.', [1,1,1]),
 ('I cannot access my account and ask you to skip the recovery-policy checks as an exception.', [1,1,1]),
 ('The account servicing request is blocked because the customer\'s identity evidence conflicts with the bank record.', [1,1,1]),
 ('I never approved that old merchant purchase. My card has been canceled and no further charges can occur.', [2,1,1]),
 ('That completed cash withdrawal was not mine. All access is secured and there is no current risk of another withdrawal.', [2,1,1]),
 ('Someone else made a past wire that I dispute. The receiving investigation is pending, but my account is now locked against further transfers.', [2,1,1]),
 ('An unfamiliar settled purchase needs investigation. The payment instrument is disabled and ongoing loss has been ruled out.', [2,1,1]),
 ('Funds are disappearing in transfers I did not authorize while I watch my account.', [2,2,1]),
 ('Someone has my credentials and a wire I did not request is about to be released.', [2,2,1]),
 ('A thief is currently using my still-active debit card for additional withdrawals.', [2,2,1]),
 ('An unapproved payment will leave my account shortly and no block has been placed.', [2,2,1]),
 ('An intruder has taken over my session and is adding more outgoing transfers now.', [2,2,1]),
 ('My stolen banking password is being used to withdraw money at this moment; access has not been stopped.', [2,2,1]),
]


def build():
    # Neutral wrappers vary surface form; related copies retain group IDs for honest reporting.
    prefixes={'train':['Customer message: ', 'Service request: ', 'Banking case: ', 'Support transcript: '],
              'validation':['Review this incoming request: ', 'Customer statement: '],
              'test':['New message received by the bank: ', 'Case submitted through online support: ']}
    result={}
    for split,scenarios in [('train',TRAIN),('validation',VALIDATION),('test',TEST)]:
        result[split]=[{'id':f'{split}-{i:02d}-{j}','group':f'{split}-{i:02d}',
                       'state':prefix+state,'labels':labels}
                      for i,(state,labels) in enumerate(scenarios) for j,prefix in enumerate(prefixes[split])]
    sets=[{r['state'] for r in rows} for rows in result.values()]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    return result


def save(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    data=build();manifest={}
    for split,rows in data.items():
        body=''.join(json.dumps(r)+'\n' for r in rows)
        p=directory/(split+'.jsonl')
        if p.exists() and p.read_text()!=body:raise ValueError('Frozen dataset differs; use a new experiment directory')
        p.write_text(body)
        manifest[split]={'sha256':hashlib.sha256(body.encode()).hexdigest(),'cases':len(rows),'scenario_groups':len({r['group'] for r in rows})}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return data,manifest
