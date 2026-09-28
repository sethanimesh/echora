"""Explicit script policy and name spelling checks; never alter the transcript."""
import re


def terms(snapshot):
    profile=(snapshot or {}).get('profile') or {}
    selection=(snapshot or {}).get('selection') or {}
    return list(dict.fromkeys([p['name'] for p in profile.get('people',[])]+profile.get('protected_terms',[])+[selection.get('recipient','')]))


def occurs(text,term):
    return bool(term and re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',text))


def policy(original,answers,language,context):
    scope=' '.join([original,context.get('conversation_reference',''),context.get('recipient','')]+[a.get('answer','') for a in answers])
    protected=[term for term in context.get('protected_terms',[]) if occurs(scope,term)]
    selected=context.get('output_script','auto')
    target=selected if selected!='auto' else ('devanagari' if re.search('[\u0900-\u097f]',original) else 'latin')
    return {'script':target if language=='Hindi/Hinglish' else 'english' if language=='English' else 'preserve',
            'protected_terms':protected}


def validate(result,policy):
    # Choices are not completed messages: each may name a different recipient
    # or use a short label. Validate the final draft after the user answers.
    if result.get('question'): return
    texts=[result['text']]
    for text in texts:
        for term in policy['protected_terms']:
            if not occurs(text,term): raise ValueError('A protected name or brand was changed or omitted')
        unprotected=text
        for term in sorted(policy['protected_terms'],key=len,reverse=True):
            unprotected=unprotected.replace(term,'')
        has_hindi=bool(re.search('[\u0900-\u097f]',unprotected))
        if policy['script'] in {'latin','english'} and has_hindi:
            raise ValueError('The requested display script was not preserved')
        if policy['script']=='devanagari' and re.search(r'[A-Za-z]',unprotected) and not has_hindi:
            raise ValueError('Hindi script was requested but not returned')


def validate_selection(text, original, answers, language, context, options):
    """Check a question option at the point it becomes a spoken message.

    A chosen recipient need not repeat the recipients in competing options.
    Exact names remain exempt from script conversion, including name-only
    answers; misspelled/case-changed names do not receive that exemption.
    """
    selected = policy(original, answers, language, context)
    known = context.get('protected_terms', [])
    present = [term for term in known if occurs(text, term)]
    rivals = [option for option in options if option != text]
    competing = {term for term in selected['protected_terms']
                 if any(occurs(option, term) for option in rivals)}
    # Only a real chosen name establishes that rival names were rejected.
    # Otherwise an omitted or transliterated sole name must not disappear.
    required = [term for term in selected['protected_terms']
                if term in present or term not in competing or not present]
    selected['protected_terms'] = list(dict.fromkeys(required + present))
    for term in known:
        if occurs(text.casefold(), term.casefold()) and not occurs(text, term):
            raise ValueError('A protected name or brand was changed')
    validate({'text': text, 'question': ''}, selected)
