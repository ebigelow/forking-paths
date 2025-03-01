
# ================================================================================================
# Prompts

# Used by: CoinFlip, LastLetter, GSM8k, HotpotQA
prompt_CoT = """\
Question:
{question}

Answer:
Let's think step by step."""


prompt_aqua = """\
Question:
{question}

Choices:
{choices}

Answer:
Let's think step by step."""
## Let's think this through step by step."""


prompt_mmlu = """\
Question:
{question}

Choices:
A) {A}
B) {B}
C) {C}
D) {D}

Answer:
Let's think step by step."""


prompt_cloze_2choice = """\
Question:
Write a story with the following constraints:
1. The story must be exactly five sentences long.
2. The story must start with the sentence: "{first_sentence}"
3. The story must end with one of the following two sentences:
  - "{last_sentence1}" 
  - "{last_sentence2}"
4. The last sentence must be exactly one of these sentences, not a rephrasing.

Answer:
{story_begin}"""


prompt_cloze_open = """\
Q: Write a story with the following constraints:
1. The story must be exactly five sentences long.
2. The story must start with the sentence: "{first_sentence}"
3. The story must be short and simple.

A: {story_begin}"""


# ================================================================================================
# Prompt formatting

def format_row_coin(row):
    s = row['inputs']

    # Strip initial "Q: __" from prompt
    s = s.replace('Q:', '')

    # Capitalize first letter of each sentence (i.e. names)
    s = '. '.join([s_[0].upper() + s_[1:] for s_ in s.split('. ')])

    # Remove extra spaces
    s = s.replace('  ', ' ').strip()

    # Rephrase "flip the coin" to "flip the coin over". The former is what the original
    #   dataset uses, but in my experience LLMs are very non-commital since they interpret "flip" to
    #   imply a random coin flip, e.g. "there's no way to know since it's a random coin"
    s = s.replace('coin.', 'coin over.')

    return prompt_CoT.format(question=s)


def format_last_letter(row):
    # Fix typos "letters" -> "letter", "words" -> "word"
    s = row['question']
    s1 = 'Take the last letters of each words in'
    s2 = 'Take the last letter of each word in'
    s = s.replace(s1, s2)

    return prompt_CoT.format(question=s)


# Used by: GSM8k, HotpotQA
def format_QA(row):
    return prompt_CoT.format(question=row['question'])


def format_aqua(row):
    choices = [o.replace(')', ') ') for o in row['options']]
    return prompt_aqua.format(
        question=row['question'], 
        choices='\n'.join(choices))


def format_mmlu(row):
    c = row['choices']
    return prompt_mmlu.format(
        question=row['question'], 
        A=c[0], B=c[1], C=c[2], D=c[3])


def format_cloze_2choice(row):
    return prompt_cloze_2choice.format(
        first_sentence=row['InputSentence1'], 
        last_sentence1=row['RandomFifthSentenceQuiz1'],
        last_sentence2=row['RandomFifthSentenceQuiz2'],
        story_begin=row['InputSentence1'], 
    )


def format_cloze_open(row):
    return prompt_cloze_open.format(
        first_sentence=row['InputSentence1'], 
        story_begin=row['InputSentence1'], 
    )


# ================================================================================================
# Extract answer formatting

# Yes/no questions (used by: CoinFlip)
messages_fn_YN = lambda full_qa_text: [{
    'role': 'user',
    'content': full_qa_text
}, {
    'role': 'user',
    'content': 'What is the final choice (Yes or No) in the Answer in the previous message?'
}, {
    'role': 'system',
    'content': 'Respond with a single-word Yes or No if possible.'
}]

therefore_YN = '\nTherefore, the answer (Yes or No) is:'


# Open-ended questions (used by: LastLetter, HotpotQA)
messages_fn_QA = lambda full_qa_text: [{
    'role': 'user',
    'content': full_qa_text
}, {
    'role': 'user',
    'content': 'What is the final answer to the Question given in the Answer in the previous message? Be brief.'
               # '\nMake sure your answer matches the type of the question, for example "Yes" or "No" if the question is binary, or the name of a specific location.'
    # 'content': 'What is the final answer to the Question given in the Answer in the previous message? Begin your message with the text "Final answer:".'
}, {
    'role': 'system',
    'content': 'Respond with only the final answer, if possible. Be brief in your response, do not include unnecessary text.'  #\n' + \
               # 'DO NOT include text such as "the final answer is" or "the answer is".'  # #  Make sure your answer matches the type of the question, for example yes/no if the question is binary, or the name of a specific location.
    # 'content': 'Be brief in your response. Do not add text such as "the answer is".'
}]

therefore_QA = '\nTherefore, the answer is:'


# Numeric answers (used by: GSM8k)
messages_fn_numeric = lambda full_qa_text: [{
    'role': 'user',
    'content': full_qa_text
}, {
    'role': 'user',
    'content': 'What is the final answer given in the Answer in the previous message?'
}, {
    'role': 'system',
    'content': 'Respond only with a number if possible. Do not include units such as "$".'
}]

therefore_numeric = '\nTherefore, the answer (arabic numerals) is:'


# Multiple choice (4 choice: MMLU)
messages_fn_ABCD = lambda full_qa_text: [{
    'role': 'user',
    'content': full_qa_text
}, {
    'role': 'user',
    'content': 'What is the final choice (A, B, C, or D) at the end of the Answer in the previous message?'
}, {
    'role': 'system',
    'content': 'Respond with a single-word multiple choice answer if possible: A, B, C or D.'
}]

therefore_ABCD = '\nTherefore, among A through D, the answer is:'


# Multiple choice (5 choice: AQuA)
messages_fn_ABCDE = lambda full_qa_text: [{
    'role': 'user',
    'content': full_qa_text
}, {
    'role': 'user',
    'content': 'What is the final choice (A, B, C, D, or E) at the end of the Answer in the previous message?'
}, {
    'role': 'system',
    'content': 'Respond with a single-word multiple choice answer if possible: A, B, C, D, or E.'
}]

therefore_ABCDE = '\nTherefore, among A through E, the answer is:'


# Story Cloze - 2 choice
def messages_fn_cloze_2choice(row):
    return lambda full_qa_text:   [{
        'role': 'user',
        'content': full_qa_text
    }, {
        'role': 'user',
        'content': 'Which of the following two sentences matches the ending of this story?'  \
                   f'\n1. "{row["RandomFifthSentenceQuiz1"]}"'  \
                   f'\n2. "{row["RandomFifthSentenceQuiz2"]}"'
    }, {
        'role': 'system',
        'content': 'Respond with a single word, either 1 or 2.'
    }]

# Note: for open-ended Story Close we use embeddings instead of answer extraction


confidence_prompt = '\nPercent confidence in final answer:'
