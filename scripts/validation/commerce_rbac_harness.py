"""Deterministic commerce/RBAC assurance harness.

Runs contract-level checks without requiring a database or payment provider.
The state machine mirrors production webhook semantics and is safe for CI.
"""
from __future__ import annotations
import argparse, hashlib, hmac, json
from dataclasses import dataclass, field
from enum import Enum

class Status(str, Enum):
    CREATED='created'; PENDING='pending'; PAID='paid'; FAILED='failed'; REFUNDED='refunded'; CHARGED_BACK='charged_back'; EXPIRED='expired'

@dataclass
class Order:
    order_id: str; user_id: str; amount: int; currency: str='COP'; status: Status=Status.CREATED
    event_ids: set[str]=field(default_factory=set); entitlements: set[str]=field(default_factory=set)

class CommerceHarness:
    def __init__(self, secret='test-secret'):
        self.secret=secret; self.orders={}; self.results=[]
    def check(self,name,ok,detail=''):
        self.results.append({'name':name,'ok':bool(ok),'detail':detail}); return ok
    def checkout(self,user,order_id,items,price_book):
        total=sum(price_book[i] for i in items if i in price_book)
        valid=len(items)>0 and all(i in price_book for i in items)
        if valid: self.orders[order_id]=Order(order_id,user,total,status=Status.PENDING)
        return valid,total
    def signature(self,event_id,order_id,status,amount,currency):
        raw=f'{event_id}{order_id}{status}{amount}{currency}{self.secret}'.encode()
        return hashlib.sha256(raw).hexdigest()
    def webhook(self,payload,signature):
        required={'event_id','order_id','status','amount','currency'}
        if not required.issubset(payload): return False,'invalid_payload'
        expected=self.signature(payload['event_id'],payload['order_id'],payload['status'],payload['amount'],payload['currency'])
        if not hmac.compare_digest(expected,signature): return False,'invalid_signature'
        order=self.orders.get(payload['order_id'])
        if not order: return False,'unknown_order'
        if payload['event_id'] in order.event_ids: return True,'duplicate_ignored'
        if payload['amount']!=order.amount or payload['currency']!=order.currency: return False,'amount_currency_mismatch'
        try: nxt=Status(payload['status'])
        except ValueError: return False,'invalid_status'
        allowed={Status.PENDING:{Status.PAID,Status.FAILED,Status.EXPIRED},Status.PAID:{Status.REFUNDED,Status.CHARGED_BACK},Status.CREATED:{Status.PENDING}}
        if nxt not in allowed.get(order.status,set()): return False,'invalid_transition'
        order.status=nxt; order.event_ids.add(payload['event_id'])
        if nxt==Status.PAID: order.entitlements.add('model:spx500_regime_gated_v1')
        if nxt in (Status.REFUNDED,Status.CHARGED_BACK): order.entitlements.clear()
        return True,'accepted'
    def rbac(self, actor, owner, action):
        roles={'free':{'read_public'},'subscriber':{'read_public','read_owned','execute_owned'},'developer':{'read_public','read_owned','publish_draft'},'admin':{'*'}}
        if actor['user_id']!=owner and action in {'read_owned','execute_owned','refund'}: return False
        return '*' in roles.get(actor.get('role',''),set()) or action in roles.get(actor.get('role',''),set())
    def run(self):
        prices={'spx500_regime_gated_v1':19900,'usdcop_regime_v1':14900}
        ok,total=self.checkout('u1','o1',['spx500_regime_gated_v1'],prices); self.check('checkout_server_total',ok and total==19900)
        p={'event_id':'e1','order_id':'o1','status':'paid','amount':19900,'currency':'COP'}
        self.check('webhook_accepts_signed_payment',self.webhook(p,self.signature(**p))[0])
        self.check('webhook_idempotent',self.webhook(p,self.signature(**p))[1]=='duplicate_ignored')
        bad=dict(p,event_id='e2',amount=1); self.check('webhook_rejects_amount_tamper',not self.webhook(bad,self.signature(**bad))[0])
        r={'user_id':'u1','role':'subscriber'}; self.check('rbac_owner_allowed',self.rbac(r,'u1','execute_owned')); self.check('bola_cross_user_denied',not self.rbac(r,'u2','execute_owned'))
        ref=dict(p,event_id='e3',status='refunded'); self.check('refund_revokes_entitlement',self.webhook(ref,self.signature(**ref))[0] and not self.orders['o1'].entitlements)
        return self.results

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args(); results=CommerceHarness().run(); out={'passed':sum(x['ok'] for x in results),'failed':sum(not x['ok'] for x in results),'checks':results}; print(json.dumps(out,indent=2) if args.json else f"commerce_rbac: {out['passed']} passed, {out['failed']} failed"); raise SystemExit(1 if out['failed'] else 0)
if __name__=='__main__': main()
