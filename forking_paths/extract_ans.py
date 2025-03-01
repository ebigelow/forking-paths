import re


def base_ans_fn(s):
    s = s.split('answer is ')[1].replace('$', '') if 'answer is ' in s else s
    s = s.strip()
    if len(s) <= 1:
        return s
    if s[-1] == '.':
        s = s[:-1]
    return s


def abcd_fn(s):
    s = s.strip()
    return s if s in ('A', 'B', 'C', 'D') else r'$\textit{Other}'

def abcde_fn(s):
    s = s.strip()
    return s if s in ('A', 'B', 'C', 'D', 'E') else r'$\textit{Other}'


def numeric_fn(s):
    pred = s
    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    
    if len(pred) > 0:
        return str(float(pred[0]))
    return s

def null_ans_fn(s):
    s = s.replace('"', '')
    s = s.replace("'", '')
    
    if len(s) <= 1:
        return s
    if s[-1] == '.':
        s = s[:-1]

    if 'answer is' in s:
        return s.split('answer is')[1]
        
    return s

def last_letter_ans_fn(s):
    s = s.lower()
    s = s.replace('"', '')
    s = s.replace("'", '')
    s = s.replace('.', '')
    if 'answer is' in s:
        s = s.split('answer is')[1]
    if 'message is' in s:
        s = s.split('message is')[1]

    s = s.replace(' ', '')
    return s.lower().strip()


ans_fn_bytask = {
    'CoinFlip': null_ans_fn,
    'LastLetter': last_letter_ans_fn,
    'AQuA': abcde_fn,
    'GSM8k': numeric_fn,
    'MMLU': abcd_fn,
    'HotpotQA': base_ans_fn,
    'StoryCloze_2Choice': numeric_fn,
    # 'StoryCloze_Open': ans_fn,
}