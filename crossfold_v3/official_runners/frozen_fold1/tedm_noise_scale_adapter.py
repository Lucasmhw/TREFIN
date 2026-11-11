"""Narrow, auditable repair for the official TEDM data-dependent noise scale.

No upstream files are modified. Install BEFORE constructing NativeModel/Noise:
  audit = ScaleAudit()
  install_noise_scale_adapter(NoiseClass, audit=audit, mode='repair')

Every finite original clipped result is preserved exactly. Only0/0 ratios receive
neutral scale1. Signed nonzero/zero quotients follow the original +/-inf clipping.
Nonfinite inputs/means are diagnosed and rejected, never silently replaced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass,field
from typing import Any,Callable
import torch


class UndefinedNoiseScaleError(RuntimeError):pass


@dataclass
class ScaleAudit:
    context:dict[str,Any]=field(default_factory=dict)
    calls:int=0
    zero_zero_elements:int=0
    zero_denominator_elements:int=0
    repaired_calls:int=0
    events:list[dict[str,Any]]=field(default_factory=list)
    max_events:int=20
    callback:Callable[[dict[str,Any],torch.Tensor,torch.Tensor],None]|None=None

    def record(self,event,y,mean):
        event={**self.context,**event}
        if len(self.events)<self.max_events:self.events.append(event)
        if self.callback:self.callback(event,y,mean)

    def summary(self):
        return {'calls':self.calls,'zero_zero_elements':self.zero_zero_elements,
                'zero_denominator_elements':self.zero_denominator_elements,'repaired_calls':self.repaired_calls,
                'events':self.events,'rule':'Only undefined0/0 -> neutral1; all finite upstream clipped values unchanged'}


def bounded_mean_ratio(mean:torch.Tensor,scale_min:float=1.,scale_max:float=8.) -> torch.Tensor:
    if not (0<scale_min<=1<=scale_max):raise ValueError('Neutral1 must lie within positive declared bounds')
    if not torch.isfinite(mean).all():raise FloatingPointError('Nonfinite cumulative mean; zero-denominator repair is inapplicable')
    denominator=mean[...,:1]
    zero_denominator=denominator==0
    # Do not evaluate an undefined quotient even in an unselected branch.
    safe_denominator=torch.where(zero_denominator,torch.ones_like(denominator),denominator)
    ratio=mean/safe_denominator
    clipped=torch.clamp(ratio,min=scale_min,max=scale_max)
    # Preserve IEEE signed-zero infinity semantics for nonzero / +/-0.
    negative=torch.logical_xor(torch.signbit(mean),torch.signbit(denominator))
    endpoint=torch.where(negative,torch.full_like(mean,scale_min),torch.full_like(mean,scale_max))
    zero_case=torch.where(mean==0,torch.ones_like(mean),endpoint)
    return torch.where(zero_denominator,zero_case,clipped)


def install_noise_scale_adapter(noise_class,*,audit:ScaleAudit|None=None,mode:str='repair'):
    if mode not in {'repair','observe','original'}:raise ValueError(mode)
    if getattr(noise_class.scale,'_trefin_noise_scale_adapter',False):
        raise RuntimeError('Scale adapter already installed in this process')
    original=noise_class.scale
    audit=audit or ScaleAudit()

    def scale(self,y,from_data=False):
        if not from_data:return original(self,y,from_data=False)
        audit.calls+=1
        if not torch.isfinite(y).all():
            audit.record({'event':'nonfinite_scale_input','shape':list(y.shape),'nonfinite_count':int((~torch.isfinite(y)).sum())},y,torch.empty(0,device=y.device))
            raise FloatingPointError('Nonfinite input reaches TEDM.scale; not repaired')
        mean=self._mean(y)
        if not torch.isfinite(mean).all():
            audit.record({'event':'nonfinite_cumulative_mean_from_finite_input','shape':list(y.shape)},y,mean)
            raise FloatingPointError('Nonfinite cumulative mean from finite input; not repaired')
        zero_denominator=mean[...,:1]==0
        zero_zero=zero_denominator&(mean==0)
        zz=int(zero_zero.sum().item())
        zd=int(zero_denominator.expand_as(mean).sum().item())
        audit.zero_zero_elements+=zz;audit.zero_denominator_elements+=zd
        if zd:
            event={'event':'zero_denominator','shape':list(y.shape),'zero_zero_elements':zz,
                   'nonzero_over_zero_elements':zd-zz,'mode':mode,'input_all_finite':True,
                   'mean_all_finite':True,'first_zero_zero_index':torch.nonzero(zero_zero,as_tuple=False)[0].tolist() if zz else None}
            audit.record(event,y,mean)
        if zz and mode=='observe':
            raise UndefinedNoiseScaleError('Confirmed finite-input0/0 in TEDM.scale before any downstream NaN')
        if mode in {'observe','original'}:
            return original(self,y,from_data=True)
        result=bounded_mean_ratio(mean,float(self.scale_min),float(self.scale_max))
        if zz:audit.repaired_calls+=1
        if not torch.isfinite(result).all():raise FloatingPointError('Scale repair did not produce a finite bounded result')
        return result

    scale._trefin_noise_scale_adapter=True
    scale._upstream_original=original
    noise_class.scale=scale
    return audit


def property_tests():
    results={}
    for dtype in [torch.float32,torch.float64]:
        generator=torch.Generator().manual_seed(104)
        y=torch.randn((7,5,113),generator=generator,dtype=dtype)
        n=torch.arange(1,114,dtype=dtype)
        mean=y.cumsum(-1)/n
        reference=torch.clamp(mean/mean[...,:1],1.,8.)
        fixed=bounded_mean_ratio(mean)
        assert torch.equal(reference,fixed)
        results[str(dtype)+'_regular_bitwise_equal']=True
    means=torch.tensor([[0.,0.,1.,-1.],[-0.,0.,1.,-1.]],dtype=torch.float64)
    fixed=bounded_mean_ratio(means)
    expected=torch.tensor([[1.,1.,8.,1.],[1.,1.,1.,8.]],dtype=torch.float64)
    assert torch.equal(fixed,expected);results['zero_zero_and_signed_infinities']=True
    mean=torch.tensor([[1e-40,1e3,-1e3]],dtype=torch.float32)
    assert torch.equal(bounded_mean_ratio(mean),torch.clamp(mean/mean[...,:1],1.,8.))
    results['finite_input_overflow_clips_identically']=True
    class Dummy:
        scale_min=1.;scale_max=8.
        @staticmethod
        def _mean(y):return y.cumsum(-1)/torch.arange(1,y.shape[-1]+1,device=y.device,dtype=y.dtype)
        def scale(self,y,from_data=False):
            if not from_data:return torch.ones_like(y)
            mean=self._mean(y);return torch.clamp(mean/mean[...,:1],1.,8.)
    audit=install_noise_scale_adapter(Dummy,mode='repair')
    y=torch.tensor([[[0.,0.,2.,-2.]]])
    v=Dummy().scale(y,from_data=True)
    assert torch.equal(v,torch.tensor([[[1.,1.,8.,1.]]]))
    assert audit.zero_zero_elements==3 and audit.repaired_calls==1
    assert torch.equal(Dummy().scale(y,from_data=False),torch.ones_like(y))
    results['actual_cumulative_zero_case_and_audit']=True
    try:Dummy().scale(torch.tensor([[[float('nan'),1.]]]),from_data=True)
    except FloatingPointError:results['nonfinite_inputs_rejected_without_replacement']=True
    else:raise AssertionError('Nonfinite input was silently masked')
    return results


if __name__=='__main__':print(json.dumps({'passed':True,'checks':property_tests()},indent=2))
