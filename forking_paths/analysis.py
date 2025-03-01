from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from forking_paths import analysis_utils


MAX_LOGPROBS = 100  ### TODO: arg to get top 100 but only filter to top K

OTHER_TOK = '$\\it{Other}$'


def answers_to_df(ans_by_ext, samples_by_ext, response_extends=None, 
                  is_chat_answer=True, ans_parse_fn=lambda s: s if s in ('A', 'B', 'C', 'D') else OTHER_TOK):
    rows = []

    response_extends = response_extends or [None] * len(ans_by_ext)
    
    for ans_samples, ex_samples, response_extend in zip(ans_by_ext, samples_by_ext, response_extends):
        (extend, idx, tok, tok_p) = response_extend or (None, -1, None, 1.0)

        for sample_res, ans_res in zip(ex_samples, ans_samples):

            # Chat object: https://platform.openai.com/docs/api-reference/chat/object
            ext_lps = sample_res['choice']['logprobs']['token_logprobs']
            finish_reason = ans_res['choice']['finish_reason']

            if finish_reason == 'content_filter':
                ans_text = ''
                ans_lps = [0]
                ans_top_lps = []
                # print('content_filter')
            elif is_chat_answer:
                choice = ans_res['choice']
                ans_text = choice['message']['content']
                ans_lps = [tok_dict['logprob'] for tok_dict in choice['logprobs']['content']] if choice.get('logprobs') else [None]
                ans_top_lps = [tok_dict['top_logprobs'] for tok_dict in choice['logprobs']['content']] if choice.get('logprobs') else None
            else:
                choice = ans_res['choice']
                ans_text = choice['text']
                ans_lps = choice['logprobs']['token_logprobs']
                ans_top_lps = choice['logprobs']['top_logprobs']

            ans_parsed = ans_parse_fn(ans_text)

            if finish_reason != 'stop':
                ans_parsed = f'Finish: {finish_reason}'

            rows.append({
                'idx': idx,                                                      # fork token index
                'tok': tok,                                                      # fork token value
                'tok_p': tok_p,                                                  # probability of the fork token being the one we used
                
                'ans': ans_parsed,                                               # cleaned answer text, e.g. `'A'` or `37.50`
                'ans_raw': ans_text,                                             # raw answer text
                'ans_lp': ans_lps[0],                                            # log prob for the (first) ans token we decoded
                'ans_lps': ans_lps,                                              # log probs for all tokens in ans sequence
                'ans_top_lps': ans_top_lps,                                      # log prob dict for each index
                
                'ext_raw': sample_res['choice']['text'],                         # raw extension text
                'ext_len': len(ext_lps),                                         # num. tokens in extension
                'ext_lp_raw': sum(ext_lps),                                      # single probability value for this extension sample
                'ext_lp': sum(ext_lps) * (1 / max(1, len(ext_lps))),             # normalized p(x) by |x|'th root - https://stats.stackexchange.com/q/603202/129261
                'ext_lps': ext_lps,                                              # list of logprobs for each extension sample token
                'ext_top_lps': sample_res['choice']['logprobs']['top_logprobs'], # logprobs dict for each ext sample token (we probably don't need this)
            })
            
    return pd.DataFrame(rows)


def normalize_tok_p(idx_tok_df):
    """normalize next-token probability"""

    # drop duplicate samples for same (index, token), then compute total prob of tokens
    sum_tok = idx_tok_df.drop_duplicates(subset=['idx', 'tok'])[['idx', 'tok_p']] \
                        .groupby(['idx'], observed=False) \
                        .sum().reset_index() \
                        .rename(columns={'tok_p': 'tok_p_sum'})
    idx_tok_df = idx_tok_df.merge(sum_tok, on=['idx'], how='left')

    idx_tok_df['tok_p'] = idx_tok_df['tok_p'] / idx_tok_df['tok_p_sum']
    return idx_tok_df


def normalize_ext_p(ans_df):
    """normalize ext_p for each (token, index) triplet"""
    groupby = ['idx', 'tok']

    sum_ext = ans_df[groupby + ['ext_lp']] \
                    .groupby(groupby, observed=True) \
                    .aggregate(logsumexp).reset_index() \
                    .rename(columns={'ext_lp': 'ext_lp_sum'})
    ans_df = ans_df.merge(sum_ext, on=groupby, how='left')

    ans_df['ext_lp'] = ans_df['ext_lp'] - ans_df['ext_lp_sum']
    return ans_df


def normalize_weighted_by_idx(idx_tok_df):
    """Re-normalize weighted histogram densities to sum to 1 for each (index, token).

    This is needed since multiplying normalized extension probs (sum to 1 for each (index, token) ) 
       and histogram values (sum=1 for (index, token) ) will result in a distribution that
       doesn't sum to 1. For the sankey plot to make sense though, we want sum ( p(answer | tok) * p(ext) ) = 1  for each (idx, token)
    """
    Z = idx_tok_df[['idx', 'tok', 'weighted']]            \
        .groupby(['idx', 'tok'], observed=False) \
        .aggregate('sum')                   \
        .reset_index().rename(columns={'weighted': 'weighted_sum'})

    idx_tok_df = idx_tok_df.merge(Z, on=['idx', 'tok'], how='left')
    idx_tok_df['weighted'] = idx_tok_df['weighted'] / idx_tok_df['weighted_sum']
    return idx_tok_df

    
def idxtok_to_idx(idx_tok_df):
    idx_df = idx_tok_df[['idx', 'ans', 'weighted']].groupby(['idx', 'ans'], observed=False).sum().reset_index()
    return idx_df


def add_answer_hist(ans_df):
    groupby = ['idx', 'tok', 'ans']

    idx_tok_df = ans_df[groupby].groupby(groupby, observed=False).size().reset_index(name='ans_count')
    idx_tok_df = idx_tok_df.merge(
        ans_df[groupby].groupby(['idx', 'tok'], observed=False).size().reset_index(name='total'),
        on=['idx', 'tok'], how='left')
    
    idx_tok_df['ans_mean'] = idx_tok_df['ans_count'] / idx_tok_df['total']
    return idx_tok_df


def ans_to_other(idx_df, ans_df, weighted_thresh=.1, max_n_ans=6, other_tok=OTHER_TOK):
    # Other for answer values with a maximum prob of < threshold
    dd = idx_df[['ans', 'weighted']].groupby(['ans']).aggregate(max).reset_index().sort_values(by='weighted')   # get max weighted probability for each answer
    keep_ans = dd[dd['weighted'] > weighted_thresh]['ans'].tolist()[-max_n_ans:]                                # filter to only top N answers with prob. above thresh

    ans_df = ans_df.copy()
    ans_df['ans'] = ans_df['ans'].apply(lambda x: other_tok if x not in keep_ans else x)
    return ans_df

def add_missing_idx_tok_combs(idx_tok_df, fill_value=0.0):
    """Add missing rows so we have values for every combination of (idx, ans)."""
    idx_tok_combs = {tuple(comb) for comb in idx_tok_df[['idx', 'tok']].to_numpy().tolist()}
    ans_set =   idx_tok_df['ans'].cat.categories.tolist() ###sorted(set(idx_tok_df['ans']))
    
    # Each answer repeated |idx_tok_combs| times
    ans_list = [[ans]*len(idx_tok_combs) for ans in ans_set]
    ans_list = np.array(ans_list).flatten().tolist()

    # Lists of all (idx, tok) combinations, repeated |ans| times
    idx_list, tok_list = zip(*(sorted(idx_tok_combs) * len(ans_set)))

    # Construct df of all outcomes/answers, for each observed (idx, tok) combination
    cols = ['idx', 'tok', 'ans']
    df_prod = pd.DataFrame(columns=cols, data=np.array([idx_list, tok_list, ans_list]).T).sort_values(by=cols)
    
    # Set column types
    df_prod['idx'] = df_prod['idx'].astype(int)
    df_prod['idx'] = pd.Categorical(df_prod['idx'], categories=idx_tok_df['idx'].cat.categories.tolist())
    df_prod['ans'] = pd.Categorical(df_prod['ans'], categories=ans_set)

    # First add token probabilities
    tok_cols = ['idx', 'tok']
    df_ret = df_prod.merge(idx_tok_df[tok_cols + ['tok_p']].drop_duplicates(tok_cols), how='left', on=tok_cols)

    # Then add extension probabilities and weighted value, if available
    cols_ = [c for c in idx_tok_df.columns if c != 'tok_p']
    df_ret = df_ret.merge(idx_tok_df[cols_], how='left', on=cols)

    # If value unavailable, i.e. we didn't observe this (idx, tok, ans) comb, substitute with `fill_value`
    fill_cols = [c for c in df_ret.columns if c not in cols]
    df_ret[fill_cols] = df_ret[fill_cols].fillna(fill_value)

    return df_ret



def get_idx_df(ans_df, tok_weighted=True, ext_weighted=True, 
               normalize_tok=True, normalize_ext=True, normalize_idx=True,
               weighted_thresh=.1, max_n_ans=6, other_tok=OTHER_TOK):
    # Wrapper function to do double conversion
    idx_df, idx_tok_df = get_idx_df_(
        ans_df, tok_weighted, ext_weighted, 
        normalize_tok, normalize_ext, normalize_idx)

    # First conversion is for this step -- converting lowest N answers to <other> token
    ans_df = ans_to_other(idx_df, ans_df, weighted_thresh, max_n_ans, other_tok)

    # Re-process final df with <other> answers substituted
    idx_df, idx_tok_df = get_idx_df_(
        ans_df, tok_weighted, ext_weighted, 
        normalize_tok, normalize_ext, normalize_idx)

    idx_tok_df = add_missing_idx_tok_combs(idx_tok_df)
    return idx_df, idx_tok_df


def get_idx_df_(ans_df, tok_weighted=True, ext_weighted=True, 
                normalize_tok=True, normalize_ext=True, normalize_idx=True):
    #
    # TODO TODO TODO   --- update to enable vector outcome repr. instead of separate rows with 'ans' column
    #
    ans_df = ans_df.copy()

    # Normalize ext probs to sum to 1 for each (token, index)
    if normalize_ext:
        ans_df = normalize_ext_p(ans_df)
        ans_df = ans_df.drop_duplicates(subset=['idx', 'tok', 'ext_raw']).reset_index(drop=True)

    # Aggregate extension probability: `sum_s p(s)` for every sample `s` for each (index, word, answer)
    groupby = ['idx', 'tok', 'ans']
    ext_df = ans_df[groupby + ['ext_lp']]        \
                    .groupby(groupby, observed=False) \
                    .aggregate(logsumexp)             \
                    .reset_index()
    
    # Compute response histograms
    idx_tok_df = add_answer_hist(ans_df)

    # Merge in token probabilities from original df
    tok_p_df = ans_df[groupby + ['tok_p']].drop_duplicates(subset=groupby)
    idx_tok_df = idx_tok_df.merge(tok_p_df, on=groupby, how='left')

    idx_tok_df = idx_tok_df.merge(ext_df, on=groupby, how='left')
    idx_tok_df['ext_p'] = np.exp(idx_tok_df['ext_lp'])

    # Normalize token probs to sum to 1 at each index
    if normalize_tok:
        idx_tok_df = normalize_tok_p(idx_tok_df)

    # Compute weighted average answer score for each (idx, token)
    tok_p = idx_tok_df['tok_p'] if tok_weighted else 1
    ext_p = idx_tok_df['ext_p'] if ext_weighted else 1
    
    idx_tok_df['weighted'] = idx_tok_df['ans_mean'] * ext_p
    if normalize_idx:    # Normalize so probs for each index sum to 1
        idx_tok_df = normalize_weighted_by_idx(idx_tok_df)
    idx_tok_df['weighted'] = idx_tok_df['ans_mean'] * tok_p

    # Convert cols to Categorical so missing values = 0 -- e.g. for a question where we have no "A" responses, A=0
    idx_tok_df['idx'] = pd.Categorical(idx_tok_df['idx'], categories=sorted(idx_tok_df['idx'].unique()))
    idx_tok_df['ans'] = pd.Categorical(idx_tok_df['ans'], categories=sorted(idx_tok_df['ans'].unique()))

    # Aggregate over tokens to compute df with indexes only
    idx_df = idxtok_to_idx(idx_tok_df)

    # Drop unnecessary columns
    idx_tok_df = idx_tok_df[['idx', 'tok', 'ans', 'ans_mean', 'tok_p', 'ext_p', 'weighted']]
    
    return idx_df, idx_tok_df


# ------------------------------------------------------------------------------------------------------


def agg_df_vector(ans_df, cols=['idx']):
    """Aggregate outcome vector mean grouped by `cols`"""

    # https://stackoverflow.com/a/58597514/4248948
    ret_df = ans_df[cols + ['ans_vector']] \
        .groupby(cols).agg(
            lambda x: np.vstack(x).mean(axis=0))

    tok_df = ans_df.drop_duplicates(cols)[cols + ['tok_p']]
    return ret_df.merge(tok_df, how='left', on=cols)


def get_idx_df_vector(ans_df, tok_weighted=True, ext_weighted=True, 
                      normalize_tok=True, normalize_ext=True, normalize_idx=True):
    # TODO TODO TODO   --- update to enable vector outcome repr. instead of separate rows with 'ans' column

    ans_df = ans_df.copy()

    # Normalize ext probs to sum to 1 for each (token, index)
    if normalize_ext:
        ans_df = normalize_ext_p(ans_df)

        # Drop duplicate samples since we're weighting by p(extension sample)
        ans_df = ans_df.drop_duplicates(subset=['idx', 'tok', 'ext_raw']).reset_index(drop=True)

    ############################################################
    ##### TODO: rewrite this for ans_df instead...
    #####
    # Normalize token probs to sum to 1 at each index
    if normalize_tok:
        idx_tok_df = normalize_tok_p(idx_tok_df)
    ############################################################

    # Add logprobs: p(extension sample) + p(token_t = w) + p()   ##### TODO: keep original token lps instead of exp'ing?
    ans_df['weight'] = 0

    if tok_weighted:
        ans_df['weight'] += np.log(ans_df['tok_p'])

    if ext_weighted:
        ans_df['weight'] += ans_df['ext_lp']

    if normalize_idx:    # Normalize so probs for each index sum to 1
        raise NotImplementedError()
        # TODO(ans_df['weight'])      ### normalize_weighted_by_idx(idx_tok_df)

    # Weighted vector representation
    ans_df['ans_vector'] = ans_df['ans_vector'] * np.exp(ans_df['weight'])

    # Aggregate over tokens to compute df with indexes only
    idx_df = agg_df_vector(ans_df, cols=['idx'])
    idx_tok_df = agg_df_vector(ans_df, cols=['idx', 'tok'])
    
    return ans_df, idx_df, idx_tok_df

