import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------
# Distance metrics

def KL_div(a, b):
    # Hand-rolled KL divergence since scipy KL is 25x slower
    #   https://datascience.stackexchange.com/a/9264/163259
    a /= a.sum()
    b /= b.sum()
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def cos_dist(a, b):
    Z = np.linalg.norm(a, 2) * np.linalg.norm(b, 2)
    sim = sum(a * b) / Z
    return 1 - sim

def lp_dist_mat(A, B, order=2):
    err = np.abs(A - B) ** order
    return (err).sum(axis=1) ** (1/order)

def KL_div_mat(A, B, eps=1e-4):
    # slightly adjust values away from 0 to avoid nan
    A = np.minimum(np.maximum(A, np.ones_like(A) * eps), 1 - (np.ones_like(A) * eps))
    B = np.minimum(np.maximum(B, np.ones_like(B) * eps), 1 - (np.ones_like(B) * eps))
    
    A /= A.sum(axis=1)[..., None]
    B /= B.sum(axis=1)[..., None]
    
    kl = A * np.log(A / B)
    kl_0 = np.where(A != 0, kl, 0)
    return kl_0.sum(axis=1)

def cos_dist_mat(A, B):
    A_norm = (A ** 2).sum(axis=1) ** .5
    B_norm = (B ** 2).sum(axis=1) ** .5
    
    Z = A_norm * B_norm
    sim = (A * B).sum(axis=1) / Z
    return 1 - sim


def get_dists(idx_df, x_col='weighted', ans_col='ans'):
    """Add distance column `d(x_0, x_t)` for each index `t`."""
    
    # Split answer column into separate columns for each categorical value
    df_ = idx_df.set_index(['idx', ans_col]).unstack([ans_col]).reset_index()
    df_.columns = [f'{ans_col}___{t}' if v == x_col else v for v,t in df_.columns]
    
    ans_cols = [c for c in df_.columns if f'{ans_col}___' in c]
    df_ = df_.sort_values(by='idx')

    # X is a matrix of all answer categorical timeseries, x_0 is a vector of values for timestep t=0
    x_0 = df_.iloc[0][ans_cols].to_numpy()[None, ...]
    X = df_[ans_cols].to_numpy()

    # Compute distances d(x_0, x_t) for each x_t
    df_['d_l1']  = lp_dist_mat(x_0, X, order=1)
    df_['d_l2']  = lp_dist_mat(x_0, X)
    df_['d_cos'] = cos_dist_mat(x_0, X)
    df_['d_kl']  = KL_div_mat(x_0, X)
    return df_


def get_entropy(x):
    x /= x.sum()
    return -np.sum(x * np.log(x))

# ----------------------------------------------------------------------------------------------------------------
# Rbeast functions

def jitter_lim(y, var=.3):
    y = y + np.random.normal(0.0, var, size=y.shape)
    y[y < 0.0] = 0.0
    y[y > 1.0] = 1.0
    return y


def get_percentile_cp(posts, percentile=.1):
    posts = np.array(posts)
    csum = np.cumsum(posts)
    return np.where(csum >= percentile)[0][0]

def get_expected_ncp(posts):
    posts = np.array(posts)
    max_cps = posts.shape[0]
    return sum(np.arange(max_cps) * posts)


# ----------------------------------------------------------------------------------------------------------------
# Forking Tokens Analysis

def get_forktok_ts(idx_tok_df, base_tokens, key_col='ans', val_col='weighted', sig_dist_thresh=None):
    idx_tok_df = idx_tok_df.copy()
    
    # Split answer column into separate columns for each categorical value
    cols = ['idx', 'tok', 'ans']
    df_ = idx_tok_df[cols + ['weighted']].set_index(cols).unstack(['ans']).reset_index()
    df_.columns = [f'ans___{t}' if v == val_col else v for v, t in df_.columns]
    
    ans_cols = [c for c in df_.columns if 'ans___' in c]
    df_ = df_.sort_values(by=['idx', 'tok']).reset_index()

    # Merge tok_p back in
    cols = ['idx', 'tok']
    df_ = df_.merge(idx_tok_df[cols + ['tok_p']].drop_duplicates(cols), how='left', on=cols)

    idxs = sorted(set(idx_tok_df['idx']))
    rows = []
    
    for idx in idxs:
        # Df for base path token
        base_tok = base_tokens[idx]
        df_i = df_[df_['idx'] == idx].copy()

        # Get token probs for all other tokens, to use for weighting
        alt_df = df_i[df_i['tok'] != base_tok]
        tp = alt_df['tok_p'].to_numpy()[..., None]

        # Get outcome / answer vectors for base token + alternate tokens
        o_alts = alt_df[ans_cols].to_numpy().astype(float)
        o_base = df_i[df_i['tok'] == base_tok].iloc[0][ans_cols].to_numpy().astype(float)[None, ...]

        # Compute distances d(w*, w_i) for each alternate token w_i relative to base token w*
        row = {
            'idx': idx,
            # Compute distance matrix for each token:   \forall_i  d(w*, w_i)
            'd_l1':  lp_dist_mat(o_base,  o_alts, order=1),
            'd_l2':  lp_dist_mat(o_base,  o_alts),
            'd_cos': cos_dist_mat(o_base, o_alts),
            'd_kl':  KL_div_mat(o_base,   o_alts),
        }

        if sig_dist_thresh is None:
            # Expected [Distance]: weighted average of distances:   \sum_i  d(w*, w_i)  p(w_i)
            row = {k: v if k == 'idx' else (v * tp).sum()
                   for k, v in row.items()}
        else:
            # Expected [d > Threshold]: Computed weighted avg of distances, with weight `1 * p(w_i)` if dist. is big enough, else 0
            row = {k: v if k == 'idx' else ((v >= sig_dist_thresh).astype(int)[..., None] * tp).sum()
                   for k, v in row.items()}
        rows.append(row)
        
    return pd.DataFrame(rows)


def get_survival_df(idx_tok_df, base_tokens, dist_fn='d_l1', d_thresholds=[0, .1, .2, .3, .5, .7]):
    # Locally import streamlit -- maybe move this fn elsewhere?
    import streamlit as st

    dfs = []

    for dist_thresh in d_thresholds:
        # Cache repeated calls to this function since it's slow to run
        thresh_df =  st.cache_data(get_forktok_ts)(
            idx_tok_df, base_tokens, sig_dist_thresh=dist_thresh)
        
        # Hazard: probability that token distribution changes by at least `d_threshold`
        thresh_df = thresh_df[['idx', dist_fn]]
        thresh_df.rename({dist_fn: 'hazard'}, axis='columns', inplace=True)
        thresh_df['threshold'] = dist_thresh

        # Cumulative product of   p(survival_t)
        #   https://stackoverflow.com/a/27912352/4248948
        thresh_df['survival'] = np.multiply.accumulate(1 - thresh_df['hazard'])
        
        dfs.append(thresh_df)

    return pd.concat(dfs)
