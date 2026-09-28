import asyncio
import json
import pytest
from communication.backend import wording_fidelity as fidelity, providers, hinglish
from communication.backend.hinglish_core.core.model import Label


def test_script_policy_uses_current_input_not_memory():
    p=fidelity.policy('bina cheeni',[], 'Hindi/Hinglish',{'conversation_reference':'मुझे चाय चाहिए।'})
    assert p['script']=='latin'
    assert fidelity.policy('चाय',[],'Hindi/Hinglish',{'output_script':'latin'})['script']=='latin'
    assert fidelity.policy('chai',[],'Hindi/Hinglish',{'output_script':'devanagari'})['script']=='devanagari'


def test_names_and_brands_must_remain_exact_and_unmentioned_names_are_not_prompted():
    p=fidelity.policy('Maya, Lipton chai',[],'Hindi/Hinglish',{'protected_terms':['Maya','Lipton','Kiran']})
    assert p['protected_terms']==['Maya','Lipton']
    fidelity.validate({'text':'Maya, mujhe Lipton chai chahiye.','question':''},p)
    with pytest.raises(ValueError):fidelity.validate({'text':'माया, mujhe Lipton chai chahiye.','question':''},p)
    with pytest.raises(ValueError):fidelity.validate({'text':'Maya, mujhe chai chahiye.','question':''},p)


def test_script_mismatch_never_silently_replaces_words():
    for script,text in [('latin','मुझे चाय चाहिए।'),('devanagari','Mujhe chai chahiye.'),('english','मुझे चाय चाहिए।')]:
        with pytest.raises(ValueError):fidelity.validate({'text':text,'question':''},{'script':script,'protected_terms':[]})
    fidelity.validate({'text':'Maya, मुझे चाय चाहिए।','question':''},{'script':'devanagari','protected_terms':['Maya']})


def test_clarification_choices_need_not_repeat_every_name():
    fidelity.validate({'text':'Maya or Kiran', 'question':'Who should receive it?',
                       'options':['Maya', 'Kiran']},
                      {'script':'devanagari','protected_terms':['Maya','Kiran']})


def test_names_are_protected_as_exact_spans_in_both_lexical_passes():
    text='Maya, HDFC Bank se chai lao. May may come.'
    lexical,queries=hinglish.lexical_pass(text,{'Maya','HDFC Bank','May'})
    result=lexical.run(hinglish.segment(text))
    labels=[(part.text,part.label) for part in result.segments if part.text.strip()]
    assert ('Maya',Label.NAME) in labels and ('HDFC',Label.NAME) in labels and ('Bank',Label.NAME) in labels
    assert ('May',Label.NAME) in labels and ('may',Label.NAME) not in labels
    assert all(q.text not in {'Maya','HDFC','Bank','May'} for q in queries)
    result,speech,_=hinglish.finish(text,lexical,[])
    assert 'Maya' in speech and 'HDFC Bank' in speech and 'May' in speech


def test_provider_reports_fidelity_failure_without_retry(monkeypatch):
    calls=[]
    async def fake(path,**kwargs):
        calls.append(kwargs['json'])
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'text':'मुझे चाय चाहिए।','question':'','options':[]})}}]}
    monkeypatch.setattr(providers,'request',fake)
    with pytest.raises(providers.ProviderFailure,match='requested script'):
        asyncio.run(providers.draft('chai',[],'Hindi/Hinglish',{'output_script':'latin'}))
    assert len(calls)==1
    assert json.loads(calls[0]['messages'][1]['content'])['fidelity']['script']=='latin'
