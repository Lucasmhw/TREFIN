"""Independent small-matrix checks of the exploratory factor-flow identities.

Numerical consistency checks complement the analytic review; they neither
select a radius nor establish a uniform guarantee by sampling.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sqrt_spd(matrix):
    values, vectors = np.linalg.eigh(matrix)
    assert values.min() > 0
    return (vectors*np.sqrt(values))@vectors.T


def speed(precision, derivative):
    covariance = np.linalg.inv(precision)
    factor = sqrt_spd(covariance)
    lengths = np.linalg.norm(factor, axis=1)
    normalized = factor/lengths[:, None]
    change = -.5*covariance@derivative@factor
    projected = (change-normalized*np.sum(normalized*change, axis=1)[:, None])/lengths[:, None]
    direct = np.linalg.norm(projected)
    relative = factor@derivative@factor
    units = factor/lengths[None]
    second = np.sum(units*(relative@relative@units), axis=0)
    first = np.sum(units*(relative@units), axis=0)
    variance = np.sqrt(max(0., np.sum(second-first**2)/4))
    np.testing.assert_allclose(direct, variance, rtol=1e-10, atol=1e-11)
    correlation = normalized@normalized.T
    return direct, relative, np.linalg.eigvalsh(correlation)[-1]


def laplacian(adjacency):
    return np.diag(np.abs(adjacency).sum(1))-adjacency


def main():
    rng = np.random.default_rng(20261118)
    n, radius = 5, .1
    edges = list(zip(*np.triu_indices(n, 1)))
    maxima = {'q_coefficient_ratio': 0., 'lambda_coefficient_ratio': 0.,
              'graph_relative_ratio': 0., 'graph_flow_ratio': 0., 'envelope_eigenvalue_ratio': 0.}
    cases = 0
    for _ in range(12):
        signs = rng.choice([-1., 1.], len(edges))
        magnitude = np.array([.03]*4+[.15]*6)
        center = np.zeros((n, n))
        for (i, j), sign, weight in zip(edges, signs, magnitude):
            center[i, j] = center[j, i] = sign*weight
        retained = np.sign(center)*np.maximum(np.abs(center)-radius/np.sqrt(2), 0)
        lower_b = rng.uniform(.03, .2, n)
        lower = laplacian(retained)+np.diag(lower_b)
        inverse_lower = np.linalg.inv(lower)
        allowed = [[-1., 1.] if i < 4 else [signs[i]] for i in range(len(edges))]

        def incidence(edge, sign):
            value = np.zeros(n)
            value[edge[0]], value[edge[1]] = 1., -sign
            return value

        envelope = np.zeros((len(edges), len(edges)))
        for e, edge in enumerate(edges):
            for f, other in enumerate(edges):
                envelope[e, f] = max((incidence(edge, se)@inverse_lower@incidence(other, sf))**2/2
                                     for se in allowed[e] for sf in allowed[f])
        largest = np.linalg.eigvalsh(envelope)[-1]
        for assignment in itertools.product(*allowed):
            graph = np.zeros_like(center)
            for (i, j), sign, weight in zip(edges, assignment, magnitude):
                graph[i, j] = graph[j, i] = sign*weight
            b = lower_b+rng.uniform(.01, .1, n)
            precision = laplacian(graph)+np.diag(b)
            assert np.linalg.eigvalsh(precision-lower).min() >= -1e-12
            incidence_matrix = np.column_stack([incidence(edge, sign) for edge, sign in zip(edges, assignment)])
            gram = (incidence_matrix.T@inverse_lower@incidence_matrix)**2/2
            assert np.max(gram-envelope) < 1e-12
            maxima['envelope_eigenvalue_ratio'] = max(maxima['envelope_eigenvalue_ratio'], np.linalg.eigvalsh(gram)[-1]/largest)
            log_q_change = rng.normal(size=n)
            actual, relative, rmax = speed(precision, np.diag(b*log_q_change))
            qcap = min(np.sqrt(rmax)/2, np.sqrt(n/8))*np.linalg.norm(log_q_change)
            maxima['q_coefficient_ratio'] = max(maxima['q_coefficient_ratio'], actual/qcap)
            log_lambda_change = rng.normal()
            actual, _, _ = speed(precision, -np.diag(b)*log_lambda_change)
            maxima['lambda_coefficient_ratio'] = max(maxima['lambda_coefficient_ratio'], actual/(np.sqrt(n)*abs(log_lambda_change)/4))
            direction = rng.normal(size=len(edges))
            derivative = incidence_matrix@np.diag(np.asarray(assignment)*direction/np.sqrt(2))@incidence_matrix.T
            actual, relative, rmax = speed(precision, derivative)
            relative_cap = np.sqrt(largest)*np.linalg.norm(direction)
            maxima['graph_relative_ratio'] = max(maxima['graph_relative_ratio'], np.linalg.norm(relative)/relative_cap)
            maxima['graph_flow_ratio'] = max(maxima['graph_flow_ratio'], actual/(min(np.sqrt(rmax)/2, np.sqrt(n/8))*relative_cap))
            cases += 1
    assert all(np.isfinite(value) and value <= 1+1e-10 for value in maxima.values())
    here = Path(__file__).resolve().parent
    report = {'status': 'passed', 'signed_region_cases': cases, 'factor_flow_identity_checks': cases*3,
              'maximum_ratios': maxima, 'source_sha256': hashlib.sha256((here/'normalized_factor_flow.py').read_bytes()).hexdigest(),
              'scope': 'Independent small-matrix consistency checks of the flow identity and coefficients; analytic proof supplies uniform validity.'}
    (here/'independent_flow_numerical_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

