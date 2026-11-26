import unittest
import torch
import torch.nn.functional as F
from experimental_answer_loss import answer_token_indices, answer_cross_entropy, mixed_loss


class AnswerLossTests(unittest.TestCase):
    def test_mask_and_causal_shift(self):
        offsets = [(0, 1), (1, 3), (3, 5), (5, 7), (0, 0)]
        self.assertEqual(answer_token_indices(offsets, 3), [2, 3])
        logits = torch.randn(5, 8, requires_grad=True)
        ids = torch.tensor([0, 1, 2, 3, 4])
        loss, n = answer_cross_entropy(logits, ids, offsets, 3, chunk_size=1)
        expected = F.cross_entropy(logits[torch.tensor([1, 2])], ids[torch.tensor([2, 3])])
        self.assertEqual(n, 2)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[1].abs().sum()), 0.0)

    def test_boundary_truncation_and_empty(self):
        offsets = [(0, 2), (2, 5), (5, 6)]
        self.assertEqual(answer_token_indices(offsets, 3), [2])
        loss, n = answer_cross_entropy(torch.randn(2, 5), torch.tensor([0, 1]), offsets[:2], 5)
        self.assertIsNone(loss)
        self.assertEqual(n, 0)

    def test_mixture_and_weights(self):
        d, c = torch.tensor(2.), torch.tensor(4.)
        self.assertEqual(mixed_loss(d, c, 0).item(), 2)
        self.assertEqual(mixed_loss(d, c, 0.25).item(), 2.5)
        self.assertEqual(mixed_loss(d, None, 0.25).item(), 2)
        with self.assertRaises(ValueError):
            mixed_loss(d, c, 1.1)


if __name__ == '__main__':
    unittest.main()
