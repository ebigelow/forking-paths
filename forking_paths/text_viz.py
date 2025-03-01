import numpy as np
import seaborn as sns


def rgb_to_htmlstr(c):
    c = [int(np.round(255*c_)) for c_ in c]
    return f'rgb({c[0]}, {c[1]}, {c[2]})'

def text_colors_html(tokens, probs, prompt=None, c_map=sns.color_palette('YlOrRd_r', as_cmap=True), 
                     print_newln=True, width='500px', height='400px', add_start_tok=''):

    if (add_start_tok is not None):
        if prompt is not None:
            add_start_tok = ' '
        if add_start_tok == '':
            probs = list(probs)[1:] + [1.0]
        else:
            tokens = [add_start_tok] + tokens
            probs = list(probs) + [1.0]
    
    # ## Normalized log probs
    # base_probs = -np.array(base_res[0]['choice']['logprobs']['token_logprobs'])
    # base_probs /= (base_probs.max() - base_probs.min())
    # -webkit-print-color-adjust: exact;   color-adjust: exact; print-color-adjust: exact;  

    # tokens = [' '] + tokens

    highlighted = ''.join([
        ('<div class="tooltip" style="display: inline;"><span style="color: black; font-size=16pt; background-color: ' +
                      rgb_to_htmlstr(c_map(prob)) + 
                      ' !important ;">' +
                  (token.replace('\n', '<br>') if print_newln else token.replace('\n', '\\n')) +      # .replace(' ', '&nbsp;')
        f'</span><span class="tooltiptext"> {t} </span></div>')
        for token, prob, t in zip(tokens, probs, range(len(tokens)))])
    
    if prompt is not None:
        # prompt_text = '<span style="color: black; background-color: #cfffd3; font-size=16pt; !important;">' + prompt + '</span>'
        prompt_text = '<span style="color: black; font-size=16pt; !important;">' + prompt + '</span>'
        highlighted = prompt_text + highlighted
    
    s = '''\
    <!DOCTYPE html>
    <html>
    
    <head>
    <style>
    span {
      -webkit-print-color-adjust: exact;
              print-color-adjust: exact;
              color-adjust: exact;        /* this + !important tag are necessary for python html -> pdf */
    }
    
    @page  {
      margin: 0;
      size: ''' + width + ' ' + height + '''
    }

    // hover tooltip for index
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        // width: 120px;
        background-color: rgba(0, 0, 0, 0.5); //black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;

        /* Position the tooltip */
        position: absolute;
        z-index: 1;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
    }


    </style>
    </head>
    
    <body>
    <div style="width: ''' + width + '''; background-color: white !important; line-height:19px;">
    ''' + highlighted + '''
    </div>
    </body>
    </html>'''

    # <div style="width: ''' + f'{width}px' if type(width) is int else width + '''; background-color: white !important; line-height:19px;">
    
    return s



def text_base_html(text, print_newln=True, width='500px', bg_color='white'):
    # prompt = base_res[0]['prompt_raw']
    # prompt = '<span style="color: black; font-size=16pt; !important;">' + prompt + '</span>'
    text = text.replace('\n', '<br>') if print_newln else text.replace('\n', '\\n')
    
    s = '''\
    <!DOCTYPE html>
    <html>
    
    <head>
    <style>
    span {
      -webkit-print-color-adjust: exact;
              print-color-adjust: exact;
              color-adjust: exact;        /* this + !important tag are necessary for python html -> pdf */
    }
    
    @page  {
      margin: 0;
      size: 520px 300px
    }
    </style>
    </head>
    
    <body>
    <div style="width: ''' + width + '''; background-color: white !important; line-height:19px;">
        <span style="color: black; font-size=16pt;  background-color: ''' + bg_color + '''; !important;">
        ''' + text + '''
        </span>
    </div>
    </body>
    </html>'''
    
    return s