from app.messaging.fidelity import preserves_bounded_facts
from app.messaging.alignment import _grounded_reading, _slot_alignment
from app.schemas import Hypothesis
from test_message_chain import chain_with


def beams(*texts):
    return [Hypothesis(id=f"h{i}", literal_text=text, sequence_score=-i, search_weight=1/len(texts)) for i,text in enumerate(texts,1)]


def test_negation_alternatives_are_not_pruned_and_keep_actual_provenance():
    evidence=beams("want tea", "not want tea")
    chain,_=chain_with({"options":[{"reading":"want tea","message":"I want tea."},{"reading":"not want tea","message":"I do not want tea."}],"note":"","unclear":False})
    result=chain._compose(evidence,"home")
    assert result.ranker.decision=="ambiguous"
    assert [m.source_hypothesis_ids for m in result.messages]==[["h1"],["h2"]]


def test_unsupported_reading_token_is_rejected_instead_of_erased():
    slots=_slot_alignment(beams("leg pain"))
    assert _grounded_reading("leg pain dinosaur",slots) is None
    assert _grounded_reading("pain",slots) is None


def test_added_medicine_body_part_changes_and_negation_are_rejected():
    assert not preserves_bounded_facts("leg pain","My leg hurts. Please bring aspirin.")
    assert not preserves_bounded_facts("left leg pain","My right arm hurts.")
    assert not preserves_bounded_facts("not want tea","I want tea.")
    assert not preserves_bounded_facts("want tea","I do not want tea.")
    assert preserves_bounded_facts("leg pain","My leg is really hurting me.")


def test_current_followup_supersedes_approved_reference():
    assert preserves_bounded_facts("without sugar","Please bring tea without sugar.",reference="Please bring tea with sugar.")
    assert not preserves_bounded_facts("without sugar","Please bring tea with sugar.",reference="Please bring tea with sugar.")


def test_approved_reference_enters_one_composition_call_without_changing_reading():
    chain,client=chain_with({"options":[{"reading":"without sugar","message":"Please bring tea without sugar."}],"note":"","unclear":False})
    result=chain._compose(beams("without sugar"),"home",conversation_reference={"text":"Please bring tea with sugar."})
    assert result.messages[0].corrected_text=="Please bring tea without sugar."
    assert result.messages[0].interpreted_intent=="without sugar"
