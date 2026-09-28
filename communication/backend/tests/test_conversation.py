import copy
import time
import pytest
from communication.backend import app, conversation, providers
from test_app import client, create, post


def approve(c,text):
    j=create(c,text);return post(c,j,'confirm',revision=j['revision'],pronunciation='original').json()


def wait(c):
    until=time.monotonic()+2
    while time.monotonic()<until:
        j=c.get('/api/session').json()['job']
        if j['status']!='drafting':return j
        time.sleep(.01)
    raise AssertionError('Draft did not finish')


def test_memory_only_sent_for_clear_followup_after_approval(client,monkeypatch):
    seen=[]
    async def draft(text,answers,lang,context):
        seen.append(context);return {'text':'Please bring me tea without sugar.','question':'','options':[]},{}
    monkeypatch.setattr(providers,'draft',draft)
    approve(client,'Please bring me tea with sugar.')
    j=create(client,'without sugar')
    post(client,j,'draft',revision=j['revision'])
    result=wait(client)
    assert seen[0]['conversation_reference']=='Please bring me tea with sugar.'
    assert result['source_text']==result['original']=='without sugar'
    assert result['conversation']['text']=='Please bring me tea with sugar.'
    assert result['confirmed'] is None
    approve(client,'Please bring me tea.')
    j=create(client,'water');post(client,j,'draft',revision=j['revision']);result=wait(client)
    assert 'conversation_reference' not in seen[-1] and result['conversation'] is None


def test_unapproved_intervening_message_breaks_chain(client,monkeypatch):
    approve(client,'Please bring tea.')
    create(client,'My leg hurts.')
    j=create(client,'without sugar')
    async def forbidden(*args): raise AssertionError('No usable reference should reach provider')
    monkeypatch.setattr(providers,'draft',forbidden)
    post(client,j,'draft',revision=j['revision']);result=wait(client)
    assert result['question'] and result['text']=='without sugar' and result['conversation'] is None


def test_scopes_expiry_and_unique_message_gate():
    snapshot={'selection':{'scenario':'home','recipient':'Meena'},'profile':{'id':'p','revision':1}}
    saved={'text':'Please bring tea.','job_id':'a','at':time.monotonic(),'scope':conversation.scope(snapshot)}
    assert conversation.reference(saved,'without sugar',snapshot)
    for field,value in [('scenario','cafe'),('recipient','Kiran'),('situation','outside')]:
        changed=copy.deepcopy(snapshot);changed['selection'][field]=value
        assert conversation.reference(saved,'without sugar',changed) is None
    assert conversation.reference({**saved,'at':time.monotonic()-601},'without sugar',snapshot) is None
    assert conversation.reference(saved,'Please bring me water.',snapshot) is None
    assert conversation.reference(saved,'without sugar',snapshot,True) is None
    assert conversation.reference(saved,'bina cheeni',snapshot)


def test_forget_removes_reference_and_inherited_draft_words(client,monkeypatch):
    async def draft(*args):return {'text':'Please bring tea without sugar.','question':'','options':[]},{}
    monkeypatch.setattr(providers,'draft',draft)
    approve(client,'Please bring tea.');j=create(client,'without sugar')
    post(client,j,'draft',revision=j['revision']);j=wait(client)
    cleared=client.post(f"/api/messages/{j['id']}/conversation",headers={'X-Echora-Client':'1'},json={'revision':j['revision'],'action':'forget'}).json()
    assert cleared['text']=='without sugar' and cleared['conversation'] is None and cleared['independent']
    s=next(iter(app.sessions.values()));assert s.memory is None and s.memory_candidate is None
    assert client.post(f"/api/messages/{j['id']}/conversation",headers={'X-Echora-Client':'1'},json={'revision':j['revision'],'action':'forget'}).status_code==409


def test_sessions_do_not_share_memory(client):
    approve(client,'Please bring tea.')
    client.cookies.clear()
    client.get('/api/session');j=create(client,'without sugar');post(client,j,'draft',revision=j['revision'])
    result=wait(client);assert result['conversation'] is None and result['question']
