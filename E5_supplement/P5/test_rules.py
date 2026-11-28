import unittest
from P5.negatives import signature,make_pairs

class Rules(unittest.TestCase):
    def test_math(self):
        self.assertEqual(signature('<<2*3=6>>'),signature('<<3*2=6>>'))
        self.assertEqual(signature('<<20*0.5=10>>'),signature('<<20-10=10>>'))
        self.assertIsNone(signature('<<1/3=0.33>>'))
        self.assertIsNone(signature('<<__import__("os")=1>>'))
    def test_train_only_pairs(self):
        rows=[dict(id=str(i),source_index=i,question=str(i)) for i in range(2000)]
        slots=[[f'<<{i}+1={i+1}>>']+['']*4 for i in range(2000)]
        a=make_pairs(rows,slots);b=make_pairs(rows,slots)
        self.assertEqual(a,b)
        self.assertTrue(all(p['donor_index']!=i and p['slots']==[0] for i,p in enumerate(a)))

if __name__=='__main__':unittest.main()
