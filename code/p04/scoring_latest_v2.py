"""Uniform GT-blind final-answer extraction for latest vLLM responses.

No question-ID exceptions, guessing, reasoning computation, unit conversion or tolerance.
"""
import re
from scoring_v2 import clean, normalize as old_normalize
VERSION='latest_vllm_final_v2'

def clean_format(text):
    text=clean(text).replace('__','')
    text=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r'\1',text)
    return text

def candidate(response):
    text=clean_format(response)
    prefix=r'(?im)^[ \t]*(?:[#>*+\-✅✔️\s]+)?(?:(?:so|therefore|thus)[,;:]?[ \t]+)?(?:the[ \t]+)?(?:(?:final|correct|best)[ \t]+)?answer(?:[ \t]+is\b[ \t]*[:=]?|[ \t]*[:=])[ \t]*([^\n]*)'
    matches=list(re.finditer(prefix,text))
    if matches:
        value=matches[-1].group(1).strip()
        if not value:
            following=[s.strip() for s in text[matches[-1].end():].splitlines() if s.strip()]
            value=following[0] if following else ''
        return value,'last_explicit_final_answer'
    boxes=re.findall(r'\\boxed\{([^{}]+)\}',text)
    if len(boxes)==1:return boxes[0].strip(),'single_boxed_answer'
    if len(boxes)>1:return None,'ambiguous_boxed_answers'
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    while lines and re.fullmatch(r'[-_*]{3,}',lines[-1]):lines.pop()
    if lines and re.fullmatch(r'\(?[A-Z]\)?[.!]?',lines[-1]):return lines[-1],'terminal_bare_letter'
    if len(lines)==1 and len(lines[0])<=120:return lines[0],'bare_short_answer'
    return None,'missing_final_answer'

def extract(response,options=None):
    options=options or {}
    value,method=candidate(response)
    if not value:return {'answer':None,'method':method,'failure':'missing_final_answer'}
    if options:
        if re.search(r'\b(?:or|and)[ \t]+(?:option[ \t]+)?\(?[A-Z]\b|\b(?:actually|instead|rather)\b|[/;]\s*\(?[A-Z]\b',value,re.I):
            return {'answer':None,'method':method,'failure':'ambiguous_candidates'}
        v=re.sub(r'^option\s+','',value,flags=re.I)
        m=re.fullmatch(r'\(?([A-Z])\)?[.!]?',v,re.I)
        if m:
            letter=m[1].upper()
            return {'answer':letter if letter in options else None,'method':method,'failure':None if letter in options else 'invalid_choice'}
        m=re.fullmatch(r'\(?([A-Z])\)?[.:)\-]\s*(.+)',v,re.I)
        if m:
            letter=m[1].upper();body=m[2]
            same=letter in options and clean_format(options[letter]).casefold().strip().rstrip('.!')==clean_format(body).casefold().strip().rstrip('.!')
            return {'answer':letter if same else None,'method':method+'_letter_and_option_text','failure':None if same else 'option_text_mismatch'}
        keys=[k for k,v in options.items() if clean_format(v).casefold().rstrip('.!')==value.casefold().rstrip('.!')]
        if len(keys)==1:return {'answer':keys[0],'method':method+'_exact_option_text','failure':None}
        return {'answer':None,'method':method,'failure':'unrecognized_choice'}
    value=re.sub(r'^\\boxed\{([^{}]+)\}$',r'\1',value)
    if method=='bare_short_answer' and (len(value.split())>8 or re.search(r'\b(?:because|think|maybe)\b',value,re.I)):
        return {'answer':None,'method':method,'failure':'missing_final_answer'}
    return {'answer':value,'method':method,'failure':None}

def normalize(value):
    value=clean_format(value)
    # Paired dollar signs are math delimiters; a single currency sign is preserved.
    if len(value)>=2 and value.startswith('$') and value.endswith('$'):value=value[1:-1]
    return old_normalize(value)

def judge(extraction,gold,options=None):
    import ast
    if extraction['answer'] is None:return False
    if options:return extraction['answer']==str(gold).strip().upper()
    if isinstance(gold,str) and gold.strip().startswith('['):
        try:
            parsed=ast.literal_eval(gold)
            if isinstance(parsed,list):gold=parsed
        except (ValueError,SyntaxError):pass
    gold=gold if isinstance(gold,list) else [gold]
    return any(normalize(extraction['answer'])==normalize(g) for g in gold)

def selftest():
    cases=[('Reasoning (A).\n### Final Answer: **B. blue**','B'),('✅ Final Answer: **C**','C'),
      ('> Final Answer: (B)','B'),('Final Answer:\n\n**C**\n---','C'),
      ('Answer: A\nCorrection.\nFinal Answer: D','D'),('Answer: B or C',None),
      ('Final Answer: B. red',None),('Answer: B actually C',None),
      ('Reasoning A.\nNo conclusion.',None),('\\boxed{B}','B'),
      ('Reasoning\nB\n---','B'),('Final Answer: option B','B')]
    opts={'A':'red','B':'blue','C':'green','D':'black'}
    assert extract('Final Answer: A. cats and dogs',{'A':'cats and dogs','B':'fish'})['answer']=='A'
    assert extract('Thus, the correct answer is B. blue',opts)['answer']=='B'
    for response,expected in cases:assert extract(response,opts)['answer']==expected,(response,extract(response,opts))
    for a,b,same in [('1/64','0.015625',True),('2 m','2 meters',True),('2 cm','2 m',False),('1.234','1.23',False),('50%','0.5',False),('$2$','2',True),('$2','2',False)]:
        assert (normalize(a)==normalize(b))==same
    return {'synthetic_cases':len(cases)+9,'failures':0,'GT_passed_to_extractor':False}
