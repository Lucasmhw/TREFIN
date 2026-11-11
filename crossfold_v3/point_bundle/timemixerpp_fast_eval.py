"""Vectorize independent-origin PyPOTS TimeMixer++ evaluation, without training changes.

The native FFT period selection pools the first tensor dimension, which contains
assets after channel-independent embedding. This implementation pools exactly the
32 assets of each origin, never different origins. Convolution/attention/mixing
operators are called unchanged, grouped only over independent batch entries.
"""
from types import MethodType
import torch
import torch.nn.functional as F


def _grouped_forward(self, x_list):
    if self.training:
        return self._original_forward(x_list)
    n_assets = self._independent_origin_assets
    rows = x_list[-1].shape[0]
    if rows % n_assets:
        raise ValueError('Channel-independent rows must be a multiple of asset count')
    origins = rows // n_assets
    xf = torch.fft.rfft(x_list[-1],dim=1)
    amplitude = xf.abs()
    # Identical reductions to native .mean(0).mean(-1), but independently for
    # every origin's contiguous block of n_assets rows.
    frequency = amplitude.reshape(origins,n_assets,*amplitude.shape[1:]).mean(1).mean(-1)
    frequency[:,0] = 0
    k = min(self.k,frequency.shape[-1])
    top = torch.topk(frequency,k,dim=-1).indices
    if (top == 0).any():
        # Preserve upstream behavior for a degenerate all-zero spectrum rather
        # than replacing its frequency rule. Such a case is not expected here.
        return [torch.cat(parts,0) for parts in zip(*[
            self._original_forward([x[b*n_assets:(b+1)*n_assets] for x in x_list])
            for b in range(origins)
        ])]
    periods = x_list[-1].shape[1] // top
    asset_amplitude = amplitude.mean(-1).reshape(origins,n_assets,-1)
    selected_amplitude = asset_amplitude.gather(2,top[:,None,:].expand(-1,n_assets,-1))
    # Preserve native top-k rank order and native stack/softmax/multiply/sum.
    # Identical-period branches may be cached, but their weights are NOT merged:
    # reassociation before LayerNorm can amplify floating-point differences.
    ranked_results = [[torch.empty_like(x) for _ in range(k)] for x in x_list]
    asset_offsets = torch.arange(n_assets,device=x_list[0].device)
    lengths = [x.shape[1] for x in x_list]
    for period in torch.unique(periods,sorted=True).tolist():
        selected_origins = torch.nonzero((periods == period).any(-1),as_tuple=True)[0]
        row_indices = (selected_origins[:,None]*n_assets + asset_offsets[None,:]).reshape(-1)
        seasons, trends = [], []
        for x in x_list:
            time_image = self.time_imaging(x.index_select(0,row_indices),int(period))
            season, trend = self.dual_axis_attn(time_image)
            seasons.append(season);trends.append(trend)
        branch = self.multi_scale_mixing(seasons,trends,lengths)
        for rank in range(k):
            local_origins=torch.nonzero(periods[selected_origins,rank] == period,as_tuple=True)[0]
            if len(local_origins)==0:continue
            local_rows=(local_origins[:,None]*n_assets+asset_offsets[None,:]).reshape(-1)
            global_rows=row_indices.index_select(0,local_rows)
            for result,value in zip(ranked_results,branch):
                result[rank].index_copy_(0,global_rows,value.index_select(0,local_rows))
    native_weights=selected_amplitude.reshape(rows,k)
    return [self.layer_norm(self.multi_reso_mixing(native_weights,x,torch.stack(ranks,dim=-1)))
            for ranks,x in zip(ranked_results,x_list)]


def install_independent_origin_eval(model,n_assets=32):
    """Patch evaluation dispatch only; all parameters, modules and training remain native."""
    if not model.channel_independence:
        raise ValueError('This implementation requires the declared channel-independent configuration')
    if model.n_features != n_assets:
        raise ValueError('Asset grouping differs from model input channels')
    for block in model.encoder_model:
        if hasattr(block,'_original_forward'):
            raise RuntimeError('Evaluation patch already installed')
        block._original_forward = block.forward
        block._independent_origin_assets = n_assets
        block.forward = MethodType(_grouped_forward,block)
    return model
