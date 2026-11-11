"""Small independent checks for loss semantics, parser ambiguity and causal sums."""
import json,torch
from data import label,teacher
from core import parse_free,INSTRUCTION
results=[]
def check(name,ok):
 assert ok,name
 results.append({'check':name,'passed':True})
check('letter label',label('B',['red','blue'])=='B')
check('letter plus actual option text',label('B. blue',['red','blue'])=='B')
check('option-only label',label('blue',['red','blue'])=='B')
check('mismatched letter and option rejected',label('A. blue',['red','blue']) is None)
check('ambiguous duplicate option rejected',label('blue',['blue','blue']) is None)
check('teacher final must match annotation',teacher('<think>evidence</think><answer>2</answer>','3') is None)
check('strict format does not accept reasoning letters',parse_free('<think>A or B</think><answer>\nB\n</answer>',['a','b'])=='B')
check('strict format rejects extra suffix',parse_free('<think>x</think><answer>B</answer> actually A',['a','b']) is None)
# Equal sample weight: token count of reasoning must not control sample weight.
ce=[torch.tensor([2.,4.,9.,1.,1.]),torch.tensor([10.,3.,3.])];regions=[{'r':[0,1],'a':[2],'f':[3,4]},{'r':[0],'a':[1],'f':[2]}]
per=[(c[z['r']].mean()+3*c[z['a']].mean()+c[z['f']].mean())/5 for c,z in zip(ce,regions)]
check('samplewise region means then batch mean',torch.allclose(torch.stack(per).mean(),torch.tensor(((3+27+1)/5+(10+9+3)/5)/2)))
logits=torch.tensor([[1.,2.,3.],[4.,1.,2.]],requires_grad=True);chosen=torch.tensor([2,0]);lp=logits.log_softmax(-1).gather(1,chosen[:,None]).sum();manual=(logits[0,2]-logits[0].logsumexp(0))+(logits[1,0]-logits[1].logsumexp(0))
check('completion logprob sum not mean',torch.allclose(lp,manual))
x=torch.tensor([-.4,-1.2,-2.],requires_grad=True);rank=x.logsumexp(0)-x[0];rank.backward();check('ranking gold gradient sign',x.grad[0]<0 and bool((x.grad[1:]>0).all()))
a=torch.tensor(0.,requires_grad=True);dpo=-torch.nn.functional.logsigmoid(.1*a);dpo.backward();check('DPO identical reference still nonzero update gradient',abs(float(dpo)-0.69314718)<1e-6 and a.grad<0)
print(json.dumps({'checks':results,'count':len(results)},indent=2))
