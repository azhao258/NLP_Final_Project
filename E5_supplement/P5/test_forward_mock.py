import ast,torch
import torch.nn.functional as F
from types import SimpleNamespace
p='P5/method.py';tree=ast.parse(open(p).read());ns={'torch':torch,'F':F,'METHOD':{'margin':.02},'_pair':({},[0,1]),'tensor_batch':lambda *a:{'ref_input_ids':torch.tensor([[2,1,0,1,2]])},'batch_positions':lambda *a:torch.tensor([[0,1,2,3,4]])}
for name in ['margin_loss','training_forward']:
 node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),p,'exec'),ns)
class C(torch.nn.Module):
 def __init__(self):super().__init__();self.e=torch.nn.Embedding(3,4);self.drop=torch.nn.Dropout(.1)
 def forward(self,input_ids,**kw):return SimpleNamespace(hidden_states=[self.drop(self.e(input_ids))])
class M(torch.nn.Module):
 def __init__(self):super().__init__();self.codi=C();self.prj=torch.nn.Linear(4,4);self.num_latent=4;self.tokenizer=None
 def forward(self,**kw):
  z=self.codi(kw['ref_input_ids']).hidden_states[-1]
  a=[self.prj(z[:,i:i+1]) for i in range(5)]
  return {'loss':sum(x.square().mean() for x in a)}
m=M().train();batch={'ref_input_ids':torch.tensor([[0,1,2,0,1]])}
torch.manual_seed(9);base=m(**batch)['loss'];base.backward();g=m.prj.weight.grad.clone();rng=torch.get_rng_state();m.zero_grad()
torch.manual_seed(9);out=ns['training_forward'](m,batch,0,0);out['loss'].backward();assert torch.equal(g,m.prj.weight.grad) and torch.equal(rng,torch.get_rng_state());m.zero_grad()
torch.manual_seed(9);out=ns['training_forward'](m,batch,0,.1);out['loss'].backward();assert torch.equal(rng,torch.get_rng_state()) and m.codi.training and torch.isfinite(m.prj.weight.grad).all()
print('PASS mocked forward: zero-factor parity, gradient, RNG and training-state preservation')
