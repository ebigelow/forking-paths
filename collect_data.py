import os
import gc
import argparse
import json
import pickle
from tqdm import tqdm

import pandas as pd    
import numpy as np
from datasets import load_dataset

import batch_prompt
from forking_paths.prompts import *
from forking_paths import (
    get_base_path, enumerate_extensions, extend_result, 
    extract_answers, extract_answers_chat, extract_answers_chat_minimal)



# ----------------------------------------
## Symbolic Reasoning  (simple)

# Coin Flip: https://huggingface.co/datasets/skrishna/coin_flip
dataset_coin = load_dataset('skrishna/coin_flip')
df_coin = dataset_coin['train'].to_pandas()

# Last Letter: https://huggingface.co/datasets/ChilleD/LastLetterConcat
dataset_letter = load_dataset('ChilleD/LastLetterConcat')
df_letter = dataset_letter['train'].to_pandas()


# ----------------------------------------
## Math Reasoning

# TinyGSM8k: https://huggingface.co/datasets/tinyBenchmarks/tinyGSM8k
# https://github.com/openai/grade-school-math
tiny_gsm = load_dataset('tinyBenchmarks/tinyGSM8k', 'main')
df_gsm = tiny_gsm['test'].to_pandas()

# https://github.com/google-deepmind/AQuA
df_aqua = pd.read_json('./domain_data/AQuA/dev.json', lines=True)

### Note: it seems like training dataset has more errors, 
###       e.g. I found a bunch of non-standard characters including symbols and emojis
# df_aqua = pd.read_json('./domain_data/AQuA/train.json', lines=True)


# ----------------------------------------
## Question Answering

# TinyMMLU: https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU
tiny_mmlu = load_dataset('tinyBenchmarks/tinyMMLU', 'all')
df_mmlu = tiny_mmlu['test'].to_pandas()


# HotpotQA: https://hotpotqa.github.io
# note: only train dataset has easy (single hop), medium (multi-hop) and hard (hard multi) - dev+test are all hard
df_hotpot = pd.read_json('./domain_data/HotpotQA/hotpot_train_v1.1.json')


# ----------------------------------------
## Story Understanding
#
# Est toks: 200 (???) toks/prompt *    ~100 branch toks/answer   *   (100 open  +  100 closed)   * 2  (sample + extract)   *  30 samples 
#    = 240 million tokens

# StoryCloze: https://www.cs.rochester.edu/nlp/rocstories/
df_cloze = pd.read_csv('./domain_data/rocstories/cloze_test_val__spring2016 - cloze_test_ALL_val.tsv', sep='\t')



"""
## Lower priority

# TinyARC: 
tiny_arc = load_dataset('tinyBenchmarks/tinyAI2_arc', 'ARC-Challenge')

"""

# Used for: AQuA, HotpotQA
def quantile_limit(df, cols):
    # TODO: test
    col_lens = {col: df[col].apply(len) for col in cols}

    for col in cols:
        lens = col_lens[col]
        df = df[ (lens > np.quantile(lens, .1)) &
                 (lens < np.quantile(lens, .9)) ]

    return df.reset_index(drop=True)



def sample_baselines(prompt, base_res, messages_fn, ans_args, 
                     therefore_str='\nTherefore, the answer is', confidence_str=confidence_prompt,
                     verbose=2, extract=2):
    if extract == 0:
        return None, None, None, None

    # Baseline 2: resample from first index (n=300   = 10 * 30)
    resample_args = {
        'temperature': 1.0,
        'n': 30,
        'max_tokens': 512
    }
    base_resamples = get_base_path(
        [prompt] * 10, resample_args, verbose,
        backend='openai')    # backend={'azure': 170, 'azure-sweden': 170, 'openai': 90})

    base_resample_ans = extract_answers_chat_minimal(prompt, base_resamples, messages_fn, ans_args)

    if extract == 1:
        return None, None, base_resamples, base_resample_ans

    # Baseline 1: "therefore the answer is" answer token logprob
    base_therefore = batch_prompt.completions(
        prompt + base_res[0]['choice']['text'] + therefore_str,
        model_args={
            'max_tokens': 15, 
            'logprobs': 100, 
            'model': 'gpt-3.5-turbo-instruct-0914',
            'temperature': 0,
            'n': 1,
        }, 
        verbose=verbose,
        queries_per_batch=1,
    )

    # Baseline 2: "Percent confidence in previous answer:"
    base_conf = batch_prompt.completions(
        prompt + base_res[0]['choice']['text'] + '\n' + therefore_str + base_therefore[0]['choice']['text'] + confidence_str,
        model_args={
            'max_tokens': 10, 
            'logprobs': 100, 
            'model': 'gpt-3.5-turbo-instruct-0914',
            'temperature': 0,
            'n': 1,
        }, 
        verbose=verbose,
        queries_per_batch=1,
    )
    return base_therefore, base_conf, base_resamples, base_resample_ans



def run_forking(fname, prompt, extract_messages_fn, therefore_str, row=None, verbose=2, extract=2):

    #########################
    # parameters
    n_samples = 30
    tok_depth = 1200
    p_thresh  = .05
    #########################
 
    base_args = {'logprobs': 20, 'max_tokens': 1200}
    extend_args = {'temperature': 1, 'n': n_samples, 'max_tokens': 512}    # Note: some extensions don't reach <eos_token>
    ans_args = {}

    print(f'Saving to: {fname}')

    # --------------------------------------------------------

    json.dump({'base_args': base_args, 'extend_args': extend_args, 'ans_args': ans_args,
               'tok_depth': tok_depth, 'p_thresh': p_thresh,       'row': json.loads(row.to_json())  },
    open(f'{fname}.json', 'w'))

    # Sample base path
    base_res = get_base_path(prompt, base_args, verbose)

    if base_res[0]['choice']['finish_reason'] != 'stop':
        print(f'ERROR for {fname}: Base result cut off before final answer')
        return

    # Extend prompt with each alternate high-prob token
    response_extends = enumerate_extensions(base_res, p_thresh, tok_depth)
    
    # Re-sample from alternate high-prob tokens
    print('\n--> Sampling extensions')
    samples_by_ext = extend_result(prompt, base_res, response_extends, extend_args, verbose)

    if extract > 0:
        # Extract answers from extensions
        print('\n--> Having chat model extract answer from the text')
        ans_by_ext = extract_answers_chat(prompt, extract_messages_fn, response_extends, samples_by_ext, ans_args, verbose)
    else:
        ans_by_ext = None

    # Sample baselines: Resample n*10 times from first index  +  "... Therefore, ..."    
    print('\n--> Sampling baselines')
    base_therefore, base_conf, base_resamples, base_resamples_ans = sample_baselines(
        prompt, base_res, extract_messages_fn, ans_args, 
        therefore_str, verbose=verbose, extract=extract)

    pickle.dump((base_res, response_extends, samples_by_ext, ans_by_ext, 
                base_therefore, base_conf, base_resamples, base_resamples_ans), 
                open(f'{fname}.pk', 'wb'))


def run_task(path, df, format_fn, extract_messages_fn, therefore_str, extract=2):
    if df.shape[0] < 1:
        print(f'No Samples for task {path.split("/")[-1]}')
        return

    for idx, row in tqdm(df.iterrows()):
        prompt = format_fn(row)

        # Eval extract fn for story cloze
        extract_messages_fn_ = extract_messages_fn(row) if extract == 1 else extract_messages_fn

        row['index'] = idx
        fname = f'{path}/{idx}'

        # Run forking paths analysis for this
        run_forking(fname, prompt, extract_messages_fn_, therefore_str, row, verbose=2, extract=extract)

        # Clear objects from RAM
        gc.collect()




def get_sampled_df(df, path, N=30):
    exist_idxs = [int(fn.split('.')[0]) for fn in os.listdir(path) if '.pk' in fn]
    keep_idxs = [i for i in df.index if i not in exist_idxs]

    N_ = N - len(exist_idxs)    
    df_samples = df.loc[keep_idxs].sample(N_)
    return df_samples


THEREFORE_MAP = {
    'CoinFlip':   therefore_YN,
    'LastLetter': therefore_QA,
    'AQuA':       therefore_ABCDE,
    'GSM8k':      therefore_numeric,
    'MMLU':       therefore_ABCD,
    'HotpotQA':   therefore_QA,
    ### 'StoryCloze_2Choice': therefore_YN,
}



TASKS = [  
    'CoinFlip',               # Symbolic reasoning
    'LastLetter',
    'AQuA',                   # Math reasoning
    'GSM8k',
    'MMLU',                   # Question answering (+ multi-hop reasoning)
    # 'HotpotQA',
    'StoryCloze_2Choice',     # Story understanding 
    ### 'StoryCloze_Open'
]




if __name__ == '__main__':

    for task in TASKS:
        # path_cost_analysis = f'./out/gpt3.5-instruct-base_path_only/{task}'
        # path = f'./out/gpt3.5-instruct-30s-300t/{task}'
        path = f'./out/gpt3.5-instruct-test_gemini_X/{task}'

        ######## TODO_GET_MISSING_BASELINE(task, path, confidence_str=confidence_prompt, verbose=2)
        os.makedirs(path, exist_ok=True)

        for _ in range(5): print('~'*120)


        if task == 'CoinFlip':
            df_coin_samples = get_sampled_df(df_coin, path, N=30)

            run_task(
                path,
                df_coin_samples,
                format_row_coin, 
                messages_fn_YN,
                therefore_YN
            )


        elif task == 'LastLetter':
            df_letter_samples = get_sampled_df(df_letter, path, N=30)
            
            run_task(
                path,
                df_letter_samples,
                format_last_letter, 
                messages_fn_QA,
                therefore_QA
            )
            

        elif task == 'AQuA':
            df_aqua['q_len'] = df_aqua['question'].apply(len)
            df_aqua['a_len'] = df_aqua['rationale'].apply(len)
            df_s_aqua = df_aqua[(df_aqua['q_len'] > np.quantile(df_aqua['q_len'], .1)) &
                                (df_aqua['q_len'] < np.quantile(df_aqua['q_len'], .9)) & 
                                (df_aqua['a_len'] > np.quantile(df_aqua['a_len'], .1)) &
                                (df_aqua['a_len'] < np.quantile(df_aqua['a_len'], .9))]
            
            df_aqua_samples = get_sampled_df(df_s_aqua, path, N=30)  # 7 from train set, now we're in dev

            run_task(
                path,
                df_aqua_samples,
                format_aqua, 
                messages_fn_ABCDE,
                therefore_ABCDE
            )


        elif task == 'GSM8k':
            df_gsm_samples = get_sampled_df(df_gsm, path, N=30)

            run_task(
                path,
                df_gsm_samples,
                format_QA, 
                messages_fn_numeric,
                therefore_numeric
            )


        elif task == 'MMLU':
            df_mmlu_samples = get_sampled_df(df_mmlu, path, N=30)

            run_task(
                path,
                df_mmlu_samples,
                format_mmlu, 
                messages_fn_ABCD,
                therefore_ABCD
            )


        elif task == 'HotpotQA':
            df_hotpot['q_len'] = df_hotpot['question'].apply(len)
            df_hotpot_ = df_hotpot[(df_hotpot['q_len'] > np.quantile(df_hotpot['q_len'], .1)) &
                                   (df_hotpot['q_len'] < np.quantile(df_hotpot['q_len'], .9))]

            df_hotpot_ = get_sampled_df(df_hotpot_, path, N=df_hotpot_.shape[0])

            df_hotpot_samples = pd.concat([
                df_hotpot_[df_hotpot_['level'] == 'easy'].sample(6),          # total = 11 - 5   (existing total files = 15, assuming 5x3)
                df_hotpot_[df_hotpot_['level'] == 'medium'].sample(6),
                df_hotpot_[df_hotpot_['level'] == 'hard'].sample(6)
            ])

            run_task(
                path,
                df_hotpot_samples,
                format_QA, 
                messages_fn_QA,
                therefore_QA
            )


        elif task == 'StoryCloze_2Choice':
            df_cloze_samples = get_sampled_df(df_cloze, path, N=30)

            run_task(
                path,
                df_cloze_samples,
                format_cloze_2choice, 
                messages_fn_cloze_2choice,
                None,
                extract=1
            )

        elif task == 'StoryCloze_Open':
            df_cloze_samples = df_cloze.sample(5)
            run_task(
                path,
                df_cloze_samples,
                format_cloze_open,
                None,
                None,
                extract=0
            )

