"""Compare pinned official preprocessing with E3; tiny-model path equivalence."""
import ast
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'e3'))
import common
from diagnose import latent_vectors


def main():
    official = ROOT/'audit_official'
    assert (official/'src/model.py').read_text(encoding='utf8') == (ROOT/'SIM-CoT/CODI/src/model.py').read_text(encoding='utf8')
    tree = ast.parse((official/'train.py').read_text(encoding='utf8'))
    main_node = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='train')
    functions = [x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_tokenize_fn']
    functions += [x for x in main_node.body if isinstance(x,ast.FunctionDef) and x.name in ('get_answer_token_position','preprocess')]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+functions,type_ignores=[])
    scope = dict(torch=torch,training_args=SimpleNamespace(remove_eos=True))
    exec(compile(ast.fix_missing_locations(module),'<pinned official preprocess>','exec'),scope)
    tok=AutoTokenizer.from_pretrained(ROOT/'assets/gpt2',use_fast=False,local_files_only=True)
    tok.add_special_tokens({'pad_token':'[PAD]'})
    rows=common.read_rows(ROOT/'e3/data/train.jsonl')
    comparisons={}
    for keep in (False,True):
        q=[r['question'] for r in rows]
        chains=[' '.join(r['cot'].split(' ') if keep else r['cot'].split(' ')[:-1]) for r in rows]
        answers=['The answer is: '+str(r['answer']).split(' ')[-1].replace('####','') for r in rows]
        with contextlib.redirect_stdout(io.StringIO()):
            official_batch=scope['preprocess'](q,chains,answers,tok,50258,50259)
        mismatches=Counter(); ids=[]
        for i,row in enumerate(rows):
            ours=common.pack(row,tok)
            fields=[]
            for name in ('encoder_input_ids','decoder_input_ids','ref_input_ids','labels','ref_labels','ref_answer_position','model_answer_position'):
                expected=official_batch[name][i]
                if isinstance(expected,torch.Tensor):expected=expected.tolist()
                current=ours[name][0]
                if expected!=current:mismatches[name]+=1;fields.append(name)
            if fields:ids.append(dict(id=row['id'],fields=fields))
        comparisons[str(keep)]=dict(mismatches=dict(mismatches),different_rows=len(ids),examples=ids[:8],
            empty_chains=sum(not c.strip() for c in chains))
    rejected=[r['id'] for r in rows if not str(r['answer']).split(' ')[-1][0].isdigit()]
    torch.set_num_threads(2)
    cfg=json.loads((ROOT/'e3/config.json').read_text());cfg.update(projection_dim=32,lora_r=4)
    training_args=common.TrainingArguments
    def cpu_args(**kw):kw.update(bf16=False,use_cpu=True);return training_args(**kw)
    def tiny(*a,**kw):return GPT2LMHeadModel(GPT2Config(vocab_size=50257,n_embd=32,n_layer=1,n_head=2,n_positions=1024))
    with patch.object(common,'TrainingArguments',cpu_args),patch('src.model.AutoModelForCausalLM.from_pretrained',tiny):
        model=common.build(cfg).eval();row=rows[0]
        projected=[];decoder_inputs=[]
        h=model.prj.register_forward_hook(lambda m,a,o:projected.append(o.detach().clone()))
        h2=model.decoder.register_forward_pre_hook(lambda m,a,kw:decoder_inputs.append(kw['inputs_embeds'].detach().clone()),with_kwargs=True)
        with torch.no_grad():out=model(**common.tensor_batch(row,model,'cpu'),step=0,step_ratio=0.)
        h.remove();h2.remove()
        zs=latent_vectors(model,row['question'],4)
        assert len(projected)==len(zs)==5
        assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(projected,zs))
        labels=common.targets(row,tok,4)
        active=[(i,s) for i,s in enumerate(labels) if s!=[tok.pad_token_id]]
        emb=model.get_embd(model.codi,model.model_name)
        for captured,(i,s) in zip(decoder_inputs,active):
            expected=torch.cat([zs[i],emb(torch.tensor([s]))],dim=1)
            assert torch.allclose(captured[:,:len(s)+1],expected,atol=1e-6,rtol=1e-5)
        # Compare answer first-token logits from training's full response vs cached free inference.
        with torch.no_grad():
            state=model.codi(input_ids=torch.tensor([tok.encode(row['question'].strip())+[model.bot_id]]),output_hidden_states=True,use_cache=True)
            for _ in range(4):
                z=model.prj(state.hidden_states[-1][:,-1:])
                state=model.codi(inputs_embeds=z,past_key_values=state.past_key_values,output_hidden_states=True,use_cache=True)
            answer=model.codi(input_ids=torch.tensor([[model.eot_id]]),past_key_values=state.past_key_values,use_cache=True)
        assert torch.allclose(out['logits'][:,0].float(),answer.logits[:,0].float(),atol=1e-5,rtol=1e-4)
    report=dict(model_source_equal_except_line_endings=True,training_rows=2000,
        official_preprocess_include_last_cot=comparisons,
        official_digit_first_answer_filter_rejects=rejected,
        tiny_path_tests=dict(five_projected_latents_match=True,decoder_embedding_prefixes_match=True,first_answer_logits_match=True),
        limitation='CPU tiny-model functional tests; not GPU numerical equivalence or proof of released-checkpoint training provenance.')
    common.write_json(official/'alignment_report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
