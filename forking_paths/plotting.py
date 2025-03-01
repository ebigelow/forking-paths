import sys

import numpy as np
import pandas as pd    
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import additive_chi2_kernel
from matplotlib import gridspec

import plotly.graph_objs as go
import plotly.offline as offline
if ('ipykernel' in sys.modules):
    offline.init_notebook_mode(connected=True)

from forking_paths.analysis import *





def format_timeseries_plot(base_tokens, idx_df, ax=None, figsize=(6, 3), dpi=None, y_log=False):
    """Change figsize and add x-labels for timeseries plots."""
    plt.rcParams['figure.figsize'] = figsize
    if dpi is not None:
        plt.rcParams['figure.dpi'] = dpi
    
    idxs = sorted(set(idx_df['idx']))
    labels = [base_tokens[i] for i in idxs]
    
    if ax is None:
        plt.xticks(rotation=45, ticks=idxs, labels=labels)
        if y_log:
            plt.yscale('log')
    else:
        ax.set_xticks(idxs)
        ax.set_xticklabels(labels, rotation=45)
        if y_log:
            ax.set_yscale('log')


def plot_categories(base_tokens, idx_df, **kwargs):
    """Plot timeseries for categorical answers, with each answer as a different color."""
    sns.lineplot(data=idx_df, x='idx', y='weighted', hue='ans')
    
    format_timeseries_plot(base_tokens, idx_df, **kwargs)
    plt.legend(loc='right')


def plot_stack(base_tokens, idx_df, normalize=True, title=None, **kwargs):
    """Plot timeseries for categorical answers, with each answer as a different color."""
    # plt.stackplot(data=idx_df, x='idx', y='weighted', hue='ans')   #### doesnt work
    if normalize:
        pt = pd.crosstab(index=idx_df['idx'], columns=idx_df['ans'], values=idx_df['weighted'], aggfunc='sum', normalize='index')  ## NORMALIZE
    else:
        pt = pd.pivot_table(idx_df, columns=['ans'], index=['idx'], values=['weighted'],   observed=False)
    pt.plot.area(lw=.05)
    
    format_timeseries_plot(base_tokens, idx_df, figsize=(10, 3))
    
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=idx_df['ans'].cat.categories.tolist(),
              bbox_to_anchor=(1.1, 1.05))   #loc='right')
    sns.despine(left=True, bottom=True)
    plt.xlim(min(idx_df['idx']), max(idx_df['idx']))
    
    mi = max(idx_df['idx'])
    x_lines_alpha = .15 if mi < 150 else .08
    plt.gca().xaxis.grid(True, color='white', alpha=x_lines_alpha, ls='-', lw=.5)
    plt.title(title)
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------
# https://plotly.com/python/sankey-diagram/
# https://plotly.com/python/parallel-categories-diagram/#basic-parallel-categories-diagram-with-counts

def plot_sankey(idx_tok_df, base_tokens, plot_idx=0, plot_n_idxs=4):
    # base_tokens = base_res[0]['choice']['logprobs']['tokens']
    
    idx_tok_df_ = idx_tok_df.copy()
    
    
    layout =  go.Layout(
        title = None,
        font = {'size': 10},
        
        xaxis = go.layout.XAxis(
            title = None,
            showticklabels=False),
        yaxis = go.layout.YAxis(
            title=None,
            showticklabels=False
        ),
    
        xaxis_range=(0,1), 
        yaxis_range=(0,1),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        showlegend=True,
    )
    
    fig = go.Figure(layout=layout)
    
    for i in range(plot_idx, plot_idx + plot_n_idxs):
        dfi = idx_tok_df_[idx_tok_df_['idx'] == i].reset_index(drop=True)
        dfi = dfi.sort_values(by='tok_p', ascending=False)    # sort tokens so highest tok_p token is first
    
        outcomes    = dfi['ans'].tolist()        # 'outcome'
        next_tokens = dfi['tok'].tolist()        # 'next_token'
        next_probs  = dfi['weighted'].tolist()   # 'weighted'
        
        prev_token = base_tokens[i-1] if i > 0 else '<START>'
        # for i, (last_token, next_tokens, next_probs) in enumerate(zip(last_tokens, next_tokens_s, next_probs_s)):
        
        out_idxs = dfi['ans'].cat.codes
        outcomes_set = dfi['ans'].cat.categories
        
        j = i - plot_idx
        pw = 1 / plot_n_idxs
        
        fig.add_trace(
            go.Parcats(
                # https://plotly.com/python/builtin-colorscales/
                line={'colorscale': 'bluyl', 'color': out_idxs,  # sunset   'color': outcome_colors * n_toks,
                      'shape': 'hspline',
                      'colorbar': {'tickvals': np.arange(len(outcomes_set)), 'ticktext': outcomes_set}  if j == 0 else None
                     },
                
                domain={'x': [pw*j, pw*(j+1)]},
                dimensions=[
                    {'label': 'Current Token',  ###prev_token,
                     'values': [prev_token] * len(outcomes)},
                     # 'values': outcomes if j == 0 else   [prev_token] * len(outcomes)},
                    {'label': 'Next Token',
                     'values': next_tokens}
                     # 'values': repeat_ls(next_tokens, n_out)}
                ],
                counts=next_probs
            )
        ) 
    
    
    ### i-1 -> i   shaded area
    for j in range(1, plot_n_idxs):
        i = j + plot_idx
        
        prev_token = base_tokens[i-1]
        prev_tok_p = idx_tok_df_[(idx_tok_df_['idx'] == i-1) & (idx_tok_df_['tok'] == prev_token)]['tok_p'].iloc[0]
        y_ = 1 - prev_tok_p
        
        '''
        fig.add_traces([
            go.Scatter(
                x=[j*pw - .03, j*pw - .02, j*pw + .02, j*pw + .03], 
                y=[y_, y_, .0, .0], 
                line={'width': 0},
                mode='lines',
                line_shape='spline',
            ),
            go.Scatter(
                x=[j*pw - .03, j*pw - .02, j*pw + .02, j*pw + .03], 
                y=[.98, .98, .99, .99], 
                line={'width': 0},
                mode='lines',
                line_shape='spline',
                
                fill='tonexty', 
                fillcolor = 'rgba(50, 50, 0, 0.1)',
            )
        ])
        '''
    fig.show()


########
# Plotly stacked plot with interactivity

def get_parcats(idx_tok_df, base_tokens, colors, plot_idx=0, plot_n_idxs=4):
    # base_tokens = base_res[0]['choice']['logprobs']['tokens']

    idx_tok_df_ = idx_tok_df.copy()

    trace_kws = []
    
    for i in range(plot_idx, plot_idx + plot_n_idxs):
        dfi = idx_tok_df_[idx_tok_df_['idx'] == i].copy()
        
        dfi['tok_codes'] = dfi['ans'].cat.codes.tolist()
        dfi = dfi.sort_values(by=['tok', 'tok_codes'], ascending=False).reset_index(drop=True)    # sort tokens so highest tok_p token is first
    
        outcomes    = dfi['ans'].tolist()        # 'outcome'
        next_tokens = dfi['tok'].tolist()        # 'next_token'
        next_probs  = dfi['weighted'].tolist()   # 'weighted'
        
        prev_token = base_tokens[i-1] if i > 0 else '<START>'
        # for i, (last_token, next_tokens, next_probs) in enumerate(zip(last_tokens, next_tokens_s, next_probs_s)):
        
        out_idxs = dfi['ans'].cat.codes.tolist()
        
        j = i - plot_idx
        pw = 1 / plot_n_idxs
        
        trace_kws.append(
            dict(
                # https://plotly.com/python/builtin-colorscales/
                line={'colorscale': [f'rgb{c}' for c in colors],        # sunset   'color': outcome_colors * n_toks,
                      'color': out_idxs,
                      'shape': 'hspline',
                     },
                
                domain={'x': [pw*j, pw*(j+1)]},
                dimensions=[
                    {'label': 'Current Token',                      ###prev_token,
                     'values': [prev_token] * len(next_tokens)},
                     # 'values': outcomes if j == 0 else   [prev_token] * len(outcomes)},
                    {'label': 'Next Token',
                     'values': next_tokens}
                     # 'values': repeat_ls(next_tokens, n_out)}
                ],
                counts=next_probs
            )
        )
    
    return trace_kws


def get_base_tokens(base_res):
    return base_res[0]['choice']['logprobs']['tokens']

    
def plotly_linked(idx_tok_df, base_tokens, plot_n_idxs=4, include_line_fig=False):
    from ipywidgets import widgets

    idx_df = idxtok_to_idx(idx_tok_df)

    # base_tokens = base_res[0]['choice']['logprobs']['tokens']
    outcomes_set = idx_df['ans'].cat.categories

    colors = sns.color_palette(palette='viridis_r', n_colors=len(outcomes_set) - 1) + [(.9, .3, .1)]

    # ============================================================================================================================================

    layout =  go.Layout(
        title = None,
        font = {'size': 10},
        
        # xaxis_range=(0,1), 
        margin=dict(l=20, r=20, t=20, b=80),
        yaxis_range=(0, 1),
        plot_bgcolor='white',
        showlegend=True,
        
        hovermode='x',  ###  'x',

        yaxis=dict(
            title='Outcome %',
            showgrid=False
        ),

        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(min(idx_df['idx']), max(idx_df['idx']))),
            ticktext = base_tokens,

            # showline=True,
            showgrid=True,

            gridwidth=.5, 
            gridcolor='rgb(.8, .8, .8)',
        )
    )

    fig = go.FigureWidget(layout=layout) 

    fig.add_traces([
        go.Scatter( 
            name = '*Other' if outcome == OTHER_TOK else outcome, 
            x = idx_df[idx_df['ans'] == outcome]['idx'], 
            y = idx_df[idx_df['ans'] == outcome]['weighted'], 
            stackgroup='one',
            fillcolor=f'rgba{colors[oi] + (.7,)}',
            line={'color': f'rgb{colors[oi]}'}
        )
        for oi, outcome in enumerate(outcomes_set)
    ])


    # ============================================================================================================================================

    layout2 =  go.Layout(
        title = None,
        font = {'size': 10},
        
        xaxis = go.layout.XAxis(
            title = None,
            showticklabels=False),
        yaxis = go.layout.YAxis(
            title=None,
            showticklabels=False
        ),

        xaxis_range=(0,1), 
        yaxis_range=(0,1),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        showlegend=False,
    )

    trace_kws = get_parcats(idx_tok_df, base_tokens, colors, plot_idx=0, plot_n_idxs=plot_n_idxs)

    fig2 = go.FigureWidget(data=[go.Parcats(**trace_kw) for trace_kw in trace_kws],
                          layout=layout2)


    # ============================================================================================================================================

    layout1 =  go.Layout(
        title = None,
        font = {'size': 10},
        
        # xaxis_range=(0,1), 
        margin=dict(l=20, r=20, t=60, b=40),
        # yaxis_range=(0, 1),
        plot_bgcolor='white',
        showlegend=True,
        hovermode='x',

        yaxis=dict(
            title='Outcome %',
            showgrid=False,
            # gridwidth=.5, 
            # gridcolor='rgb(.8, .8, .8)',
        ),

        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(min(idx_df['idx']), max(idx_df['idx']))),
            ticktext = base_tokens,

            showgrid=True,
            gridwidth=.5, 
            gridcolor='rgb(.8, .8, .8)',
        )
    )

    fig1 = go.FigureWidget(layout=layout1) 

    fig1.add_traces([
        go.Scatter( 
            name = outcome, 
            x = idx_df[idx_df['ans'] == outcome]['idx'], 
            y = idx_df[idx_df['ans'] == outcome]['weighted'], 
            # stackgroup='one',

            line={'color': f'rgb{colors[oi]}'}
        )
        for oi, outcome in enumerate(outcomes_set)
    ])

    # ============================================================================================================================================


    def update_point(trace, points, selector):
        """https://plotly.com/python/click-events/"""
        if len(points.xs) > 0:
            plot_idx = min(points.xs[0], max(idx_tok_df['idx']) - plot_n_idxs)
            
            with fig.batch_update():
                trace_kws = get_parcats(idx_tok_df, base_tokens, colors, plot_idx=plot_idx, plot_n_idxs=plot_n_idxs)
        
                for i in range(plot_n_idxs):
                    fig2.data[i].dimensions[0].values = trace_kws[i]['dimensions'][0]['values']    # current token
                    fig2.data[i].dimensions[1].values = trace_kws[i]['dimensions'][1]['values']    # next token
                    fig2.data[i].counts = trace_kws[i]['counts']
                
        
    for trace in fig.data + fig1.data:
        trace.on_click(update_point)

    figs = [fig1, fig, fig2] if include_line_fig else [fig, fig2]
    return figs, widgets.VBox(figs)
