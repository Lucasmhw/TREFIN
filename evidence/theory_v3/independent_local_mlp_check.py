"""Independent small-dimensional check of uniform MLP state bounds.

This numerical validation is separate from the analytic proof and is not
used to choose the constants. It deliberately includes masked coordinates,
symmetric edge coordinates, pooled heads, and signed graph perturbations.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from local_mlp_bounds import scalar_mlp_state_bound


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    n, k = 4, 3
    edges = torch.triu_indices(n, n, 1)
    mask = torch.tensor([1., 0., 1.])
    rc, ra = .2, .1
    ratios = {'concept_jacobian': 0., 'graph_jacobian': 0., 'argument_interval': 0.}
    cases = 0
    for seed in range(8):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(2*k+2, 7), nn.SiLU(), nn.Linear(7, 5), nn.SiLU(), nn.Linear(5, 1))
        net.requires_grad_(False)
        c = torch.randn(n, k)
        a = torch.randn(n, n)*.1
        a = (a+a.T)/2
        a.fill_diagonal_(0.)
        context = torch.randn(2)

        def unpack(vector):
            cc = vector[:n*k].reshape(n, k)
            aa = vector.new_zeros(n, n)
            aa[edges[0], edges[1]] = vector[n*k:]/math.sqrt(2)
            aa = aa+aa.T
            return cc, aa

        def features(cc, aa):
            return torch.cat([cc*mask, (aa@cc)*mask, context.expand(n, -1)], dim=-1)

        center = torch.cat([c.flatten(), math.sqrt(2)*a[edges[0], edges[1]]])
        for pooled in (False, True):
            def target(vector):
                cc, aa = unpack(vector)
                ff = features(cc, aa)
                if pooled:
                    ff = ff.mean(0, keepdim=True)
                return net(ff).flatten()

            bound, interval = scalar_mlp_state_bound(net, features(c, a)[None], c, a, rc, ra, mask, pooled)
            center_value = target(center).detach()
            for _ in range(12):
                dc = torch.randn(n*k)
                dc *= rc/dc.norm()
                da = torch.randn(len(edges[0]))
                da *= ra/da.norm()
                point = center+torch.cat([dc, da])
                jac = torch.autograd.functional.jacobian(target, point, vectorize=True)
                jc = torch.linalg.matrix_norm(jac[:, :n*k], ord=2)
                ja = torch.linalg.matrix_norm(jac[:, n*k:], ord=2)
                difference = (target(point)-center_value).abs()
                iratio = (difference/interval).max()
                current = [float(jc/bound[0]), float(ja/bound[1]), float(iratio)]
                for name, value in zip(ratios, current):
                    ratios[name] = max(ratios[name], value)
                    if value > 1+1e-10:
                        raise AssertionError((seed, pooled, name, value))
                cases += 1
    here = Path(__file__).resolve().parent
    result = {'passed': True, 'cases': cases, 'network_seeds': 8,
              'pooled_and_nonpooled': True, 'concept_radius': rc, 'graph_radius': ra,
              'mask': mask.tolist(), 'maximum_observed_to_bound_ratios': ratios,
              'source_sha256': hashlib.sha256((here/'local_mlp_bounds.py').read_bytes()).hexdigest(),
              'scope': 'Independent numerical consistency check; analytic inequalities establish uniform validity.'}
    (here/'independent_local_mlp_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
