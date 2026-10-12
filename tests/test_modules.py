import unittest
import torch
from torch import nn
from figra import (FiGRA, FeatureAdapter,
                      KinematicGraphRefiner, make_optimizer, train_step)


class TestLifter(nn.Module):
    """Synthetic interface fixture, not an implementation of a pose backbone."""
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(2,512)
        self.head=nn.Linear(512,3)

    def forward(self, tracks):
        return self.head(self.projection(tracks[:,:, :, :].mean(1)))[:,None]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_parameter_counts(self):
        modules=[FeatureAdapter(),KinematicGraphRefiner()]
        self.assertEqual([sum(p.numel() for p in m.parameters()) for m in modules],
                         [67553,17347])

    def test_zero_initialization(self):
        f=torch.randn(2,1,17,512)
        torch.testing.assert_close(FeatureAdapter()(f)[0],f,rtol=0,atol=0)
        pose=torch.randn(2,1,17,3)
        torch.testing.assert_close(KinematicGraphRefiner()(pose)[0],pose-pose[:,:,0:1])

    def test_stage2_freezing_and_kgr_gradient(self):
        m=FiGRA(TestLifter(),stage='stage2')
        self.assertFalse(any(p.requires_grad for p in m.backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.fam.parameters()))
        self.assertEqual(sum(p.numel() for p in m.adaptation_parameters()),17347)
        m.train()
        self.assertFalse(m.backbone.training)
        self.assertFalse(m.fam.training)
        y=m(torch.randn(2,1,9,17,2))
        y.square().sum().backward()
        self.assertGreater(m.kgr.output_projection.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in m.fam.parameters()))
        self.assertFalse(m.backbone.head._forward_pre_hooks)

    def test_arms_and_budget(self):
        for stage,arm,n in [('stage1','fam',67553),('stage2','figra',17347),('joint','figra',84900)]:
            m=FiGRA(TestLifter(),arm=arm,stage=stage)
            self.assertEqual(sum(p.numel() for p in m.adaptation_parameters()),n)
            self.assertEqual(m(torch.randn(2,2,9,17,2)).shape,(2,2,17,3))
        for width in [11,22,32,53,74]:
            m=FiGRA(TestLifter(),fam_width=width,stage='joint')
            self.assertEqual(sum(p.numel() for p in m.adaptation_parameters()),18884+2063*width)

    def test_save_restore(self):
        m=FiGRA(TestLifter())
        n=FiGRA(TestLifter())
        n.backbone.load_state_dict(m.backbone.state_dict())
        n.load_adapter_state(m.adapter_state())
        x=torch.randn(2,1,9,17,2)
        torch.testing.assert_close(m(x),n(x),rtol=0,atol=0)

    def test_training_steps_preserve_frozen_tensors(self):
        x=torch.randn(2,1,9,17,2)
        batch={'detector':x,'target':torch.randn(2,1,17,3)*100}
        for stage,arm in [('stage1','fam'),('joint','figra'),
                          ('stage2','figra')]:
            m=FiGRA(TestLifter(),arm=arm,stage=stage)
            frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
            value=train_step(m,batch,make_optimizer(m),torch.Generator().manual_seed(7))
            self.assertTrue(torch.isfinite(torch.tensor(value['loss'])))
            for n,p in m.named_parameters():
                if n in frozen: torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)

    def test_person_independence_and_pelvis(self):
        m=FiGRA(TestLifter())
        nn.init.normal_(m.fam.output.weight, std=0.02)
        nn.init.normal_(m.kgr.output_projection.weight, std=0.02)
        x=torch.randn(2,2,9,17,2)
        y=m(x)
        x[:,1]=torch.randn_like(x[:,1])*3
        z=m(x)
        torch.testing.assert_close(y[:,0],z[:,0],rtol=0,atol=0)
        torch.testing.assert_close(y[:,:,0],torch.zeros_like(y[:,:,0]),rtol=0,atol=0)

    def test_stage1_gradient_and_hook_cleanup(self):
        m=FiGRA(TestLifter(),arm='fam',stage='stage1')
        m(torch.randn(2,1,9,17,2)).square().mean().backward()
        self.assertGreater(m.fam.output.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        self.assertFalse(m.backbone.head._forward_pre_hooks)

    def test_invalid_stage(self):
        for arm,stage in [('fam','stage2'),('figra','stage1'),('other','joint')]:
            with self.assertRaises(ValueError): FiGRA(TestLifter(),arm=arm,stage=stage)


if __name__=='__main__': unittest.main()
