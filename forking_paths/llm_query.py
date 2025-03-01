import sys

import numpy as np
import pandas as pd    
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logsumexp

from sklearn.metrics.pairwise import additive_chi2_kernel
from matplotlib import gridspec

import plotly.graph_objs as go
import plotly.offline as offline
if ('ipykernel' in sys.modules):
    offline.init_notebook_mode(connected=True)

import batch_prompt


MAX_LOGPROBS = 100  ### TODO: arg to get top 100 but only filter to top K

OTHER_TOK = '$\\it{Other}$'


def enumerate_extensions(res, p_thresh, tok_depth):
    """Extend prompt with each alternate high-prob token

    Returns:
        list[tuple] : [( extended string, index, token=w, token prob p(w | index) ) 
                       for each (index, word)]
        """
    lp = res[0]['choice']['logprobs']
    response_extends = []
    
    for tok, lps, idx in zip(lp['tokens'], lp['top_logprobs'], range(tok_depth)):
        ps = {tok_: np.exp(p) for tok_, p in lps.items() 
              if ((np.exp(p) > p_thresh) or (tok_ == tok))}
    
        for tok_, p in ps.items():
            extend = ''.join(lp['tokens'][:idx] + [tok_])
            response_extends.append((extend, idx, tok_, p))
    
    return response_extends


def extend_result(prompt, res, response_extends, model_args, verbose=2):
    """Greedily/stochastically sample from alternate high-prob tokens"""
    n_e = len(response_extends)

    model_args_ = {
        'max_tokens': 512, 
        'logprobs': 5, 
        # 'model': 'gpt-35-instruct-0914',  
        'model': 'gpt-3.5-turbo-instruct-0914',
        'temperature': 0.0,
        'n': 1
    }
    model_args_.update(**model_args)

    # extend_prompts = [prompt + e for e, _, _, _ in response_extends]
    extended = batch_prompt.completions(
        prompt + '{extend}',
        [{'extend': e} for e, _, _, _ in response_extends],
        model_args=model_args_, 
        verbose=verbose,
        queries_per_batch=1,
        backend={'azure': 170, 'azure-sweden': 170, 'openai': 90}    ### TODO
    )

    samples_by_ext = [[r for r in extended if r['prompt_args']['extend'] == e]
                      for e, _, _, _ in response_extends]
    return samples_by_ext



def extract_answers(prompt, extract_prompt, response_extends, samples_by_ext, model_args, verbose=2, format_fn=None):
    """Extract answers from extensions  - TODO deprecated, using chat to extract instead"""
    # For efficiency, we batch our queries together in a flat list
    if format_fn is None:
        format_fn = lambda prompt, ext, r: {'prompt': prompt, 'response': ext[0] + r['choice']['text']}

    # This dict records the index in this list for each (token_idx, extension_idx)
    prompt_args = {(tok_i, ext_i): format_fn(prompt, x, r) \
                   for tok_i, (x, res) in enumerate(zip(response_extends, samples_by_ext)) \
                   for ext_i, r in enumerate(res)}

    p_idx, p_args = zip(*prompt_args.items())

    model_args_ = {
        'max_tokens': 10, 
        'logprobs': 10, 
        'model': 'gpt-3.5-turbo-instruct-0914',
        'temperature': 0,
        'n': 1,
    }
    model_args_.update(model_args)
    
    extend_answers = batch_prompt.completions(
        extract_prompt, 
        prompt_args=p_args,
        model_args=model_args_, 
        verbose=verbose,
        queries_per_batch=10,
    )

    p_idx_to_ans = dict(zip(p_idx, extend_answers))

    ans_by_ext = [[p_idx_to_ans[(tok_i, ext_i)] for ext_i, _ in enumerate(res)]
                  for tok_i, res in enumerate(samples_by_ext)]
    return ans_by_ext



def extract_answers_chat(prompt, messages_fn, response_extends, samples_by_ext, model_args, verbose=2, backend='google'):
    """Extract answers from extensions, using chat interface to save >2x on completion tokens"""

    # This dict records the index in this list for each (token_idx, extension_idx)
    prompt_args = {(tok_i, ext_i): messages_fn(prompt + x[0] + r['choice']['text'])   \
                   for tok_i, (x, res) in enumerate(zip(response_extends, samples_by_ext)) \
                   for ext_i, r in enumerate(res)}

    p_idx, messages = zip(*prompt_args.items())

    model_args_ = {
        'max_tokens': 20, 
        'logprobs': True,
        'top_logprobs': 20,
        'model': 'google/gemini-1.5-flash-001',    # 'gpt-3.5-turbo-0125',     # 'gpt-3.5-turbo-0125',
        'temperature': 0,
        'n': 1,
    }
    model_args_.update(model_args)

    extend_answers = batch_prompt.chat_completions(
        messages, 
        model_args=model_args_, 
        verbose=verbose,
        backend=backend,   #'openai'   #{'azure': 1, 'azure-sweden': .7, 'openai': 1.5}
    )

    p_idx_to_ans = dict(zip(p_idx, extend_answers))

    ans_by_ext = [[p_idx_to_ans[(tok_i, ext_i)] for ext_i, _ in enumerate(res)]
                  for tok_i, res in enumerate(samples_by_ext)]
    return ans_by_ext



def extract_answers_chat_minimal(prompt, sample_results, messages_fn, model_args, verbose=2, backend='google'):
    # TODO TODO   - simplified function for extracting answers from resample baseline

    # This dict records the index in this list for each (token_idx, extension_idx)
    messages = [messages_fn(prompt + r['choice']['text']) for r in sample_results]

    model_args_ = {
        'max_tokens': 20, 
        'logprobs': True,
        'top_logprobs': 20,
        'model': 'google/gemini-1.5-flash-001',    # 'gpt-3.5-turbo-0125',     # 0125, 0613
        'temperature': 0,
        'n': 1,
    }
    model_args_.update(model_args)

    extend_answers = batch_prompt.chat_completions(
        messages, 
        model_args=model_args_, 
        verbose=verbose,
        backend=backend,
    )
    return extend_answers


def resps_to_text(ress):
    return [res['choice']['text'] for res in ress]


def get_base_path(prompt, model_args, verbose=2, **kwargs):
    model_args_ = {
        'max_tokens': 256, 
        'logprobs': 10, 
        'model': 'gpt-3.5-turbo-instruct-0914',
        'temperature': 0,
    }
    model_args_.update(model_args)
    
    base_res = batch_prompt.completions(
        prompt, 
        model_args=model_args_, 
        verbose=verbose,
        **kwargs
    )
    return base_res


def sample_paths(prompt, extract_messages_fn=None,
                 p_thresh=.01, tok_depth=30, 
                 verbose=2,
                 base_args={}, extend_args={}, ans_args={}):
    # Sample initial path, everything else is computed relative to this
    print('\n--> Sampling base result - greedily decoded')
    base_res = get_base_path(prompt, base_args, verbose)

    if base_res[0]['choice']['finish_reason'] != 'stop':
        raise ValueError('Base result cut off before final answer')

    # Extend prompt with each alternate high-prob token
    response_extends = enumerate_extensions(base_res, p_thresh, tok_depth)
    
    # Greedily sample from alternate high-prob tokens
    print('\n--> Sampling extensions')
    samples_by_ext = extend_result(prompt, base_res, response_extends, extend_args, verbose)

    # Extract answers from extensions
    print('\n--> Having GPT extract answer from the text')
    ans_by_ext = extract_answers_chat(prompt, extract_messages_fn, response_extends, samples_by_ext, ans_args, verbose)
    # ans_by_ext = extract_answers(prompt, extract_prompt, response_extends, samples_by_ext, ans_args, verbose, extract_format_fn)

    return base_res, response_extends, samples_by_ext, ans_by_ext