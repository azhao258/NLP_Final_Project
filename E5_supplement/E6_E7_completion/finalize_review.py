"""Attach explicit AI case-review notes and package-readable E6/E7 reports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

NOTES = {
'gsm8k-test-194': ('应为20-5×2=10。', '前缀遮蔽答对，但解释没有5×2和20-10；其中2×3=4自身算术错误。答案改善未伴随解释改善。'),
'gsm8k-test-52': ('漫画重30×0.25=7.5，剩余7.5磅需移走7.5/0.5=15个玩具。', '参考标记最后一项为15=15，未完整表达除法，属于参考标注不完整；gold=15可独立复算。各模型输出的小乘法均未解释本题。'),
'gsm8k-test-442': ('40+3×5-(30/10)×2=49。', 'v1在槽0及槽3分别匹配3×5=15、3×2=6，但链条整体不完整且答案错。改进模型失去部分局部匹配；不能把局部模板偶合当作完整推理。'),
'gsm8k-test-1205': ('4×3-(1+4+3)=4。', 'P1答案由8变4，解释仍反复输出2×3=6，未展示总量减去前三项的计算。'),
'gsm8k-test-39': ('按数据集四倍速度解释，跑速6、走速1.5；跑2小时、走4小时，总路程18。', '改进模型首槽匹配3×2=6或交换顺序2×3=6；其余步骤未完成，答案仍错。参考有6个<<>>标记，尾部两式合并到槽4，另有走速计算未置于标记内。局部匹配提升不等于完整解题。'),
'gsm8k-test-666': ('Kassidy=90，Rayna=150，(120+90+150)/3=120。', '双方答案都对但生成的小乘法不对应这些数量或步骤，属于答案与解释脱节。'),
'gsm8k-test-473': ('两条鱼长12英寸=1英尺；1/10×100%=10%。', 'P2/P3/P4答案正确但解释不对应单位换算和比例。参考标记包含1=1且最终百分比未标入<<>>，应披露参考提取不完整；不改变gold或原评分。'),
'gsm8k-test-242': ('(5-2)×2×5×6/60=3小时。', 'v1槽1的3×2=6与参考偶合，P3/P4在该位置不再匹配；双方最终答案仍错。'),
'gsm8k-test-720': ('本周3+5=8，比上周多8-4=4。', 'P4/P5从4变8；8等于本周总量，但不能仅凭输出认定内部确实漏减。解释也没有支持正确的加减链。'),
'gsm8k-test-798': ('Billy时薪11.5、Sally时薪10.5，差额×20=20。', 'P5从8变20，但五槽均为2×3=6。参考最后的230-210未被<<>>标记捕获；题目答案可独立核算。'),
'gsm8k-test-955': ('假设无出院，1000-1000/5-50×14=100。', 'P5从100变8；解释不涉及病床、初始占用或两周新增人数。'),
'gsm-hard-train-257': ('1+3+0.5+1.5+2=8小时。', '无发布参考链；这里是AI独立题意核算，不计入步骤gold。P5从8变4，其乘法模板无法解释各项时间总和。'),
'gsm-hard-train-1051': ('原耗时38+2=40秒。官方36对应40×0.9；严格速度提升10%应为40/1.1≈36.36秒。', '题意与官方gold存在速度/耗时表述歧义。P3/P5命中官方36仍按官方正确计分，但不能将此例称为无争议的语义推理改善。无参考链，不补造步骤评分。'),
'chal-389': ('10-2=8。', 'P3或P5丢失正确答案；解释乘法与减法无关。SVAMP只有整式，逐槽参考匹配不适用。'),
'chal-959': ('青蛙39-19=20，老鼠20-12=8。', 'P5从12变8，但解释没有两次减法；属于答案改善、解释未改善。'),
'chal-239': ('昨天工作9/3=3分钟，今天44件是无关信息。', 'P3由36变3，但解释没有9/3；答案改善不等于表达式解释改善。')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--p5',type=Path,required=True);args=ap.parse_args()
    out=args.results
    rows=lambda p:[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
    dump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
    cases=rows(out/'case_review.jsonl');pairs=rows(out/'paired_questions.jsonl')
    for ds,mode in [('gsm_hard','gained'),('svamp','gained'),('svamp','lost')]:
        r=next(r for r in pairs if r['model']=='p3' and r['dataset']==ds and r['answer_outcome']==mode)
        if not any((c['model'],c['id'])==(r['model'],r['id']) for c in cases):cases.append(dict(r,case_id=f'C{len(cases)+1:02}',selection_reasons=['P3_OOD_case'],review_status='AI_PRELIMINARY_HUMAN_PENDING',human_reviewer='',human_agrees='',human_notes=''))
    for r in cases:r['ai_recalculation'],r['ai_review']=NOTES[r['id']]
    with (out/'case_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in cases)
    (out/'case_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases),encoding='utf8')
    text=['E7 配对案例 AI 初审（不是人类已签字的人工核验）','选例：按答案得失、步骤变化及答案/解释脱节目的性抽样；覆盖六种改进，含P5全部7个答案得失案例、P3 OOD案例。不是随机样本，不能估计全体错误比例。','未改gold、未剔除题目、未修改原评分。请在case_review.csv填写human_reviewer、human_agrees、human_notes。','']
    for r in cases:
        text += [f"{r['case_id']} | {r['model']} | {r['id']} | {','.join(r['selection_reasons'])}",r['question'],f"官方gold：{r['gold']}；v1：{r['v1_prediction']} → 改进：{r['prediction']}",f"原参考：{r['reference']}",f"v1解码：{r['v1_explanations']}",f"改进解码：{r['explanations']}",f"保守兼容槽位：{r['v1_conservative_equivalent_count']} → {r['conservative_equivalent_count']}",r['ai_recalculation'],r['ai_review'],'人工确认：待填写','']
    (out/'CASE_REVIEW.txt').write_text('\n'.join(text),encoding='utf8')
    summary=rows(out/'summary.jsonl');groups=rows(out/'problem_types.jsonl')
    report=['E6/E7 补齐报告（2026-11-28）','', 'E6：已完成七模型统一离线统计，21行模型×数据集总表、3600条逐题对照、204行分组比较。','E7：已完成21000槽位评分、105行槽位统计、189行重叠错误分类，以及34个目的性配对案例的AI初审。人类审核签字尚待A/B完成。','原六模型所有汇总字段回归检查通过，未改写B的原结果。新增P5。','', '答案正确数（每集200题）：']
    for m in ['v1','prefix_mask','p1','p2','p3','p4','p5']:
        rs=[next(r for r in summary if r['model']==m and r['dataset']==ds) for ds in ['gsm8k','gsm_hard','svamp']]
        report.append(f"{m}: GSM8K {rs[0]['correct']}, GSM-Hard {rs[1]['correct']}, SVAMP {rs[2]['correct']}; 合计{sum(r['correct'] for r in rs)}/600")
    report+=['','E6 P5分题型发现（沿用冻结标签；分组重叠，禁止跨轴累加）：']
    for r in groups:
        if r['model']=='p5' and r['axis'] in ['operation_family','reference_steps','native_type']:
            report.append(f"{r['dataset']} / {r['axis']}={r['label']}: n={r['n']}, 正确{r['v1_correct']}→{r['correct']}, 新增{r['gained']}/丢失{r['lost']}, 小组={r['small_group']}")
    report += ['', 'E7结论：P5 GSM8K严格匹配3/634、兼容9/634；v1为4/634、8/634。P5全部测试3000槽中2870槽为2×3=6或3×2=6（95.67%），模板问题未解决。', 'P3/P5在GSM-Hard新增命中1051题存在“速度提高10%”与官方“时间减少10%”的歧义。保持官方准确率，同时披露该案例；不能把它当作确定的语义改善。', '参考标记质量：52题的15=15、473题的1=1，以及39/798等题的未标记计算，使自动提取链不等于完整自然语言推理。未据此改分。', '', '效率与压缩：七模型均K=4，五个解释槽含最后readout-only槽。参考长度比不代表速度比。summary.csv含平均/中位/P95历史延迟，未进行七模型同场重复计时，不能声称加速。', 'GSM-Hard参考链缺失：对齐指标为N/A。SVAMP只有整式：整式兼容与逐步匹配区分。槽位自身算术成立不等于题意对应；错误标签可重叠。', '统计边界：单seed、小测试子集；v1曾使用测试结果选型；全部属于探索性比较。未做显著性或因果忠实性证明。', '', '下一步交付：保留v1冻结基线。A/B阅读CASE_REVIEW.txt并在CSV签字；选择代表案例用于报告/海报；保留失败结果。无需为了完成本轮E6/E7重训。若声称稳定速度收益才需新增同场计时，该项当前未声称。', '本包不含模型权重；P5最终checkpoint必须另行备份。训练总GPU租赁成本仍需结合全部会话和平台账单，不能用单次训练循环耗时替代。']
    (out/'REPORT.txt').write_text('\n'.join(report),encoding='utf8')
    coverage=[]
    for m in ['prefix_mask','p1','p2','p3','p4','p5']:
        local=[r for r in cases if r['model']==m]
        available={tag:any(tag in r['selection_reasons'] for r in local) for tag in ['answer_gain','answer_loss','step_gain','step_loss','correct_answer_incomplete_explanation']}
        coverage.append(dict(model=m,cases=len(local),coverage=available,note='prefix_mask无兼容步骤提升候选，不虚构成功案例' if m=='prefix_mask' else 'AI初审，人工待签'))
    dump(out/'case_coverage.json',coverage)
    dump(out/'complete.json',dict(status='AUTOMATED_ANALYSIS_AND_AI_REVIEW_COMPLETE_HUMAN_SIGNOFF_PENDING',models=7,questions_per_model=600,paired_questions=3600,scored_slots=21000,case_reviews=len(cases),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
    print('Completed',len(cases),'AI-reviewed cases; human sign-off remains pending')

if __name__=='__main__':main()
