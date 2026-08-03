from scripts.validation.commerce_rbac_harness import CommerceHarness

def test_commerce_rbac_contracts_pass():
    results=CommerceHarness().run()
    assert all(r['ok'] for r in results), results

def test_invalid_transition_rejected():
    h=CommerceHarness(); h.checkout('u','o',['x'],{'x':10})
    p={'event_id':'e','order_id':'o','status':'refunded','amount':10,'currency':'COP'}
    assert h.webhook(p,h.signature(**p)) == (False,'invalid_transition')
