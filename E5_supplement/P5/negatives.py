"""Conservative numeric filtering. Never loads validation or test rows."""
import ast
from fractions import Fraction
import random
import re


def value(text):
    if len(text)>500:raise ValueError('too long')
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            return Fraction(ast.get_source_segment(text,n))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            v=visit(n.operand);return -v if isinstance(n.op,ast.USub) else v
        if isinstance(n,ast.BinOp):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
        raise ValueError('unsupported')
    return visit(ast.parse(text,mode='eval').body)


def signature(text):
    try:
        equations=re.findall(r'<<([^<>]+)>>',text)
        if not equations or re.sub(r'<<[^<>]+>>','',text).strip():return None
        results=[]
        for equation in equations:
            a,b=equation.split('=');left,right=value(a),value(b)
            if left!=right:return None
            results.append(str(right))
        return results
    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):return None


def make_pairs(rows,slots):
    assert len(rows)==2000 and [r['source_index'] for r in rows]==list(range(2000))
    sig=[[signature(s) if s else None for s in seq] for seq in slots]
    result=[]
    for i,row in enumerate(rows):
        candidates=list(range(len(rows)));random.Random(405105+i).shuffle(candidates)
        picked=None
        for j in candidates:
            if i==j or ''.join(row['question'].lower().split())==''.join(rows[j]['question'].lower().split()):continue
            valid=[k for k in range(5) if sig[i][k] and sig[j][k]
                   and len(sig[i][k])==len(sig[j][k])
                   and set(sig[i][k]).isdisjoint(sig[j][k])]
            if valid:
                picked=dict(anchor_id=row['id'],donor_id=rows[j]['id'],donor_index=j,slots=valid,
                    positive=[slots[i][k] for k in valid],negative=[slots[j][k] for k in valid]);break
        result.append(picked or dict(anchor_id=row['id'],donor_id=None,donor_index=None,slots=[]))
    return result
