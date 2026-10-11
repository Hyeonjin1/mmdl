"""P04 primary scorer v1, frozen using independent synthetic cases, never GT-guided extraction."""
import re,unicodedata
from fractions import Fraction
VERSION='p04_final_v1'

def clean(text):
    return unicodedata.normalize('NFKC',str(text)).replace('**','').replace('`','').replace('−','-').strip()

def final_candidate(response):
    text=clean(response)
    pattern=r'(?im)^\s*(?:#{1,6}\s*)?(?:(?:therefore|thus|so)[,;:]?\s*)?(?:(?:the\s+)?final\s+)?answer\s*(?::|is\b)\s*([^\n]*)'
    matches=list(re.finditer(pattern,text))
    if matches:
        match=matches[-1];candidate=match.group(1).strip()
        if not candidate:
            remaining=text[match.end():].strip().splitlines()
            candidate=remaining[0].strip() if remaining else ''
        return candidate,'last_explicit_answer'
    boxed=re.findall(r'\\boxed\{([^{}]+)\}',text)
    if len(boxed)==1:return boxed[0].strip(),'single_boxed_answer'
    if len(boxed)>1:return None,'ambiguous_boxed_answers'
    if '\n' not in text and len(text)<=100:return text,'bare_short_answer'
    return None,'missing_final_answer'

def extract(response,options=None):
    options=options or []
    value,method=final_candidate(response)
    if not value:return dict(answer=None,method=method,failure='missing_final_answer')
    if options:
        letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(options)]
        if re.search(r'\b(?:or|and)\b|[/;]\s*\(?[A-Z]\b',value,re.I):
            return dict(answer=None,method=method,failure='multiple_candidates')
        match=re.fullmatch(r'\(?([A-Z])\)?(?:[.:\-]\s*.*|\s+.*|[.!])?',value,re.I)
        if match and match.group(1).upper() in letters:
            if len(re.findall(r'\([A-Z]\)',value))>1:return dict(answer=None,method=method,failure='multiple_candidates')
            return dict(answer=match.group(1).upper(),method=method,failure=None)
        matching=[chr(65+i) for i,opt in enumerate(options) if clean(opt).casefold().rstrip('.')==value.casefold().rstrip('.')]
        if len(matching)==1:return dict(answer=matching[0],method=method+'_option_text',failure=None)
        return dict(answer=None,method=method,failure='unrecognized_choice')
    if method=='bare_short_answer' and (len(value.split())>8 or re.search(r'\b(?:because|therefore|think|maybe)\b',value,re.I)):
        return dict(answer=None,method=method,failure='missing_final_answer')
    return dict(answer=value,method=method,failure=None)

UNITS={'centimeter':'cm','centimeters':'cm','metre':'m','metres':'m','meter':'m','meters':'m','degrees':'°','degree':'°','percent':'%','dollars':'$','dollar':'$'}
def normalize(value):
    value=clean(value).casefold().strip('$') if str(value).strip().startswith('\\(') else clean(value).casefold()
    value=value.replace('\\(','').replace('\\)','').replace('\\%','%')
    value=re.sub(r'\s+',' ',value).strip().rstrip('.')
    value=re.sub(r'\\(?:d?frac)\{([+-]?\d+)\}\{(\d+)\}',r'\1/\2',value)
    value=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',value)
    m=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?:/\d+)?)\s*([a-z°%²³$]*)',value)
    if m:
        try:return ('number',Fraction(m.group(1)),UNITS.get(m.group(2),m.group(2)))
        except (ValueError,ZeroDivisionError):pass
    return ('text',value)

def judge(extracted,gold,options=None):
    if extracted['answer'] is None:return False
    answers=gold if isinstance(gold,list) else [gold]
    if options:return any(extracted['answer']==str(a).strip().upper() for a in answers)
    return any(normalize(extracted['answer'])==normalize(a) for a in answers)

def selftest():
    cases=[('Reasoning (A) is tempting.\n**Final Answer:** (B) correct','B'),('### Answer:\n(C)','C'),('Answer: A\nCorrection.\nFinal answer: D','D'),('(B)','B'),('Final answer: B. choice text','B'),('Answer: B or C',None),('Answer: (A) and (B)',None),('Considering A and B.\nNeither is justified.',None),('Answer: Z',None),('Reasoning mentions C\n\\boxed{D}','D'),('\\boxed{A} then \\boxed{B}',None),('Answer: blue','B'),('Answer: B / C',None),('A. red','A')]
    options=['red','blue','green','black']
    for response,expected in cases:assert extract(response,options)['answer']==expected,(response,extract(response,options),expected)
    for a,b,same in [('1,000','1000',True),('1/2','0.5',True),('2 meters','2 m',True),('2 cm','2 m',False),('50%','0.5',False),('1.234','1.23',False),('−3','-3',True),('3.0','3',True)]:assert (normalize(a)==normalize(b))==same,(a,b)
    return dict(synthetic_cases=len(cases)+8,GT_passed_to_extractor=False,version=VERSION)
