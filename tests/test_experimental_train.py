import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import train_distill as td

class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(6, 8))
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits[None, :input_ids.shape[1]])

class ToyTokenizer:
    def __call__(self, text, **kwargs):
        n = min(len(text), kwargs['max_length'])
        return {'input_ids': torch.tensor([[i % 8 for i in range(n)]]),
                'attention_mask': torch.ones(1, n, dtype=torch.long),
                'offset_mapping': torch.tensor([[(i, i + 1) for i in range(n)]])}

class ToySpine:
    names = ['teacher']
    ids = {'teacher': np.arange(8)}
    def __len__(self): return 8

class ToyTeachers:
    names=['teacher']
    spine=ToySpine()
    def logits(self, text, max_len): return None
    def fused_target(self, *args, **kwargs):
        return [np.log(np.full((4, 8), 1/8, dtype=np.float32))]

class TrainingIntegration(unittest.TestCase):
    def test_answer_loss_changes_update_but_not_baseline(self):
        models = []
        for weight in (0., 0.5):
            model=ToyStudent(); teachers=ToyTeachers()
            with tempfile.TemporaryDirectory() as d, patch.object(td, '_shared_offsets', return_value=[[1,2,3,4]]):
                hist=td.train_student(teachers, teachers.spine, ToyTokenizer(), model,
                    [{'prompt':'ab','solution':'cdef'}], steps=1, k=8, T=1., lr=.01,
                    max_len=6, out_dir=d, save_every=1, answer_weight=weight, log=lambda _:None)
                self.assertEqual(len(hist),1)
                self.assertIsNotNone(td.latest_ckpt(d))
            models.append(model.logits.detach().clone())
        self.assertFalse(torch.allclose(models[0], models[1]))
        self.assertTrue(torch.isfinite(models[1]).all())

if __name__=='__main__': unittest.main()
