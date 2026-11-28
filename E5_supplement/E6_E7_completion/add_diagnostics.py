"""Recompute retrieval counts from row exports; keep train/test/validation separate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probes',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    rows=[];refs=[];hashes={};base=None
    for model in ['v1','p3','p4','p5']:
        d=a.probes/model
        cfg=json.loads((d/'config.json').read_text());summary=json.loads((d/'summary.json').read_text())
        if base is None:base=cfg
        assert all(cfg[k]==base[k] for k in ['ids','precision','data_sha256','source_sha256'])
        records=[json.loads(x) for x in (d/'retrieval.jsonl').read_text().splitlines()]
        hashes[model]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file()}
        for scope in ['same_question','other_question_same_slot']:
            for group in ['all','consumed','readout']:
                rs=[r for r in records if r['scope']==scope and (group=='all' or r['consumed']==(group=='consumed'))]
                old=summary['retrieval'][scope+'/'+group];n=len(rs);hits=sum(r['unique_top1'] for r in rs)
                assert old['n']==n and old['unique_top1']==(hits/n if n else None)
                rows.append(dict(model=model,scope=scope,group=group,n=n,top1_count=hits,top1=hits/n if n else None,mean_margin=mean(r['margin'] for r in rs) if n else None))
        for kind,value in summary['reference'].items():refs.append(dict(model=model,path=kind,**value))
    for name,rs in [('retrieval_comparison',rows),('reference_path',refs)]:
        with (a.output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    (a.output/'diagnostic_provenance.json').write_text(json.dumps(dict(split='validation',n=200,input_hashes=hashes,warning='Gold-conditioned diagnostic; no causal faithfulness or free-answer accuracy claim. No representation tensors packaged.'),indent=2),encoding='utf8')
    (a.output/'RETRIEVAL_NOTES.txt').write_text('验证集专项诊断：v1/P3/P4/P5采用相同200题、候选集合规模、精度和脚本。\nP5同题命中194/437，相比v1的187/437净增7；跨题同槽位7/478，相比6/478仅增1。平均跨题间隔仍为负，不能称为稳定对应能力改善。P4的绝对正例余弦提升不等于正负区分改善。\n各模型参考表示来自自身网络，比较不是冻结参考编码器的纯latent消融。不同题候选可能存在语义等价/歧义；检索不是语义真值，也不是因果忠实性证明。\nP5完整验证/测试对应性和答案指标未获得整体收益，保留负结果。\n',encoding='utf8')
    print('PASS retrieval row counts and summaries')

if __name__=='__main__':main()
