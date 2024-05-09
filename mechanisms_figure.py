import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

beta = 0.6
payoff_size = 0.45

#constants
p2_star = 0.6
p2_starstar = 0.3
p1_star = 0.53 
fig_height = 2
fig_width = 1.5
params = {'p2_star': p2_star, 'p2_starstar': p2_starstar, 
          'p1_star': p1_star, 
          'fig_height': 2, 'fig_width': 1.5, 'payoff_size': payoff_size, 'beta': beta}
    

def draw_inter_payoff_cell(ax, bottom_left, facecolor, info, width, height = payoff_size/2, params=params):
    p2_star, p2_starstar, p1_star, fig_height, fig_width, payoff_size, beta = params['p2_star'], params['p2_starstar'], params['p1_star'], params['fig_height'], params['fig_width'], params['payoff_size'], params['beta']
    edgecolor = 'grey'
    textcolor = 'blue'
    delta = 0.01
    x1, y1 = bottom_left
    h, w = height, width*payoff_size/2
    ax.add_patch(patches.Rectangle((x1, y1), w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=1))
    if 'q' in info:
        q = info['q']
        ax.add_patch(patches.Rectangle((x1, y1), w, q*h, edgecolor=edgecolor, linewidth=1, hatch='//', fill=False))
    else: 
        q =0
        ax.add_patch(patches.Rectangle((x1, y1+q*payoff_size/2), w, (1-q)*h, (1-q)*payoff_size/2, edgecolor=edgecolor, linewidth=1, hatch='//', fill=False))
    if 'tq' in info:
        t = info['tq']
        ax.text(x1 + w/2, y1 + q*h/2, f'${t}$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    if 'tn' in info:
        t = info['tn']
        ax.text(x1 + w/2, y1 + q*h + (1-q)*h/2, f'${t}$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    if 'c' in info:
        t = info['c']
        c_color = 'red'
        ax.plot([x1, x1], [y1 + q*h, y1 + h], color=c_color)
        ax.text(x1-delta, y1 + q*h + 1/2*(1-q)*h, f'${t}$', horizontalalignment='right', verticalalignment='center', color = c_color)
    

def draw_payoff_cell(ax, bottom_left, info, params=params):
    p2_star, p2_starstar, p1_star, fig_height, fig_width, payoff_size, beta = params['p2_star'], params['p2_starstar'], params['p1_star'], params['fig_height'], params['fig_width'], params['payoff_size'], params['beta']
    x1, y1 = bottom_left
    x1_beta = x1 + payoff_size/2*(1-beta)
    info_u = info['u']
    if 1 in info_u:
        u1, u2 = info_u[1], info_u[2]
        p1, p2 = u1['p'], u2['p']
        draw_inter_payoff_cell(ax, (x1, y1), 'white', u1, 1-beta, height = p1*payoff_size/2, params=params)
        draw_inter_payoff_cell(ax, (x1, y1+p1*payoff_size/2), 'white', u2, 1-beta, height = (1-p1)*payoff_size/2, params=params)
    else:
        draw_inter_payoff_cell(ax, (x1, y1), 'white', info['u'], 1-beta, height = payoff_size/2, params=params)
    draw_inter_payoff_cell(ax, (x1_beta, y1), 'yellow', info['s'], beta, height = payoff_size/2, params=params)
    ax.add_patch(patches.Rectangle((x1, y1), payoff_size/2, payoff_size/2, fill=False, edgecolor='black', linewidth=1))
    if 'b' in info:
        t = info['b']
        c_color = 'red'
        ax.plot([x1_beta, x1+payoff_size/2], [y1+payoff_size/2, y1 + payoff_size/2], color=c_color)
        ax.text(x1_beta + 1/2*beta*payoff_size/2, y1+payoff_size/2, f'${t}$', horizontalalignment='center', verticalalignment='bottom', color = c_color)
    

def draw_payoffs(ax, bottom_left, info, params=params):
    p2_star, p2_starstar, p1_star, fig_height, fig_width, payoff_size, beta = params['p2_star'], params['p2_starstar'], params['p1_star'], params['fig_height'], params['fig_width'], params['payoff_size'], params['beta']
    x1, y1 = bottom_left
    x2, y2 = x1 + payoff_size, y1 + payoff_size
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    draw_payoff_cell(ax, (x1, y1), info['l1']['l2'], params=params)
    draw_payoff_cell(ax, (x1, ym), info['l1']['h2'], params=params)
    draw_payoff_cell(ax, (xm, y1), info['h1']['l2'], params=params)
    draw_payoff_cell(ax, (xm, ym), info['h1']['h2'], params=params)
    textcolor, lambdacolor, lambdafintsize = 'black', 'darkblue', 8
    delta, lambdadelta = 0.01, 0.013
    ax.text(x1 + payoff_size/4, y1-delta, f'$l_1$', horizontalalignment='center', verticalalignment='top', color = textcolor)
    ax.text(x1 + 3*payoff_size/4, y1-delta, f'$h_1$', horizontalalignment='center', verticalalignment='top', color = textcolor)
    if 'Lambda1' and info['Lambda1'] != '':
        ax.text(x1 + 3*payoff_size/4, y1+payoff_size+ lambdadelta, f'${info['Lambda1']}$', horizontalalignment='center', verticalalignment='bottom', color = lambdacolor, fontsize = lambdafintsize)
    ax.text(x1-delta, y1 + payoff_size/4, f'$l_2$', horizontalalignment='right', verticalalignment='center', color = textcolor)
    ax.text(x1-delta, y1 + 3*payoff_size/4, f'$h_2$', horizontalalignment='right', verticalalignment='center', color = textcolor)
    if 'Lambda2' in info and info['Lambda2'] != '':
        ax.text(x1+payoff_size+ lambdadelta, y1 + 3*payoff_size/4, f'${info['Lambda2']}$', horizontalalignment='left', verticalalignment='center', color = lambdacolor, rotation=90, fontsize = lambdafintsize)


def draw_cell(ax, bottom_left, top_right, info, label, params=params):
    p2_star, p2_starstar, p1_star, fig_height, fig_width, payoff_size, beta = params['p2_star'], params['p2_starstar'], params['p1_star'], params['fig_height'], params['fig_width'], params['payoff_size'], params['beta']
    x1, y1 = bottom_left
    x2, y2 = top_right
    width = fig_width*(x2 - x1)
    height = fig_height*(y2 - y1)
    ax.add_patch(patches.Rectangle((fig_width*x1, fig_height*y1), width, height, edgecolor='black', linewidth=1, fill=False))
    draw_payoffs(ax, (fig_width*x1 + (width-payoff_size)/2, fig_height*y1 + (height - payoff_size)/2), info, params=params)
    labelcolor, delta = 'green', 0.05, 
    center = (x1*fig_width+delta, y2*fig_height-delta)
    ax.text(*center, label, horizontalalignment='center', verticalalignment='center', color = labelcolor)
    ax.add_patch(patches.Circle(center, 0.04, edgecolor=labelcolor, fill=False))

def Mechanism_a2_h1_l2(params=params):
    name = 'Mechanism_a2_h1_l2'
    params = params.copy()
    params['beta'] = 0.45
    p2_star, p2_starstar, p1_star, fig_height, fig_width, payoff_size, beta = params['p2_star'], params['p2_starstar'], params['p1_star'], params['fig_height'], params['fig_width'], params['payoff_size'], params['beta']
    #draw a box and divide it with three horizontal lines
    fig, ax = plt.subplots(figsize=(5*fig_width, 4*fig_height))
    info = {'l1':{'l2':{'s':{'q':0.6, 'c':'y_A', 'tn':'l_1'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-l_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': r'p_2-\left(1-p_2\right)\frac{p_2^*}{1-p_2^*}'}
    draw_cell(ax, (0, p2_star), (p1_star, 1), info=info, label='A', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-l_2'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0, 'tn':'h_2-l_2'}},},
            'Lambda1': 'p_1', 'Lambda2': r'p_2-\left(1-p_2\right)\frac{p_2^{**}}{1-p_2^{**}}'}
    draw_cell(ax, (0, p2_starstar), (p1_star, p2_star), info=info, label='B', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'h_1'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}, 'b':'x'}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':0.4, 'c':'y_B', 'tq':'-h_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': r'p_2-\left(1-p_2\right)\frac{p_2^{**}}{1-p_2^{**}}'}
    draw_cell(ax, (p1_star, p2_starstar), (1, 1), info=info, label='C', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': '0'}
    draw_cell(ax, (0, 0), (1, p2_starstar), info=info, label='D', params=params)

    textcolor, delta = 'black', 0.03
    ax.text(fig_width+delta, p2_star*fig_height, f'$p_2^*$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(fig_width+delta, p2_starstar*fig_height, '$p_2^{**}$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(0-delta, 0-delta, f'$0$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(p1_star*fig_width, fig_height+delta, f'$p_1^*$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.set_xlabel('$p_1$', fontsize=14)
    ax.set_ylabel('$p_2$', fontsize=14)

    #set boundaries of the figure at fig_width and fig_height
    ax.set_xlim([0, fig_width])
    ax.set_ylim([0, fig_height])
    #remove ticks and labels on axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #show the figute
    plt.savefig(f'figures/{name}.png', dpi=600)

def Mechanism_a1_h1_l2(params=params):
    name = 'Mechanism_a1_h1_l2'
    params = params.copy()
    params['beta'] = 0.45
    p2_star, p2_starstar, p1_star, fig_height, fig_width, payoff_size, beta = params['p2_star'], params['p2_starstar'], params['p1_star'], params['fig_height'], params['fig_width'], params['payoff_size'], params['beta']
    #draw a box and divide it with three horizontal lines
    fig, ax = plt.subplots(figsize=(5*fig_width, 4*fig_height))
    info = {'l1':{'l2':{'s':{'q':0.6, 'c':'y_A', 'tn':'l_1'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-h_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': r'p_2-\left(1-p_2\right)\frac{p_2^*}{1-p_2^*}'}
    draw_cell(ax, (0, p2_star), (p1_star, 1), info=info, label='A', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}},},
            'Lambda1': 'p_1', 'Lambda2': r'p_2-\left(1-p_2\right)\frac{p_2^{**}}{1-p_2^{**}}'}
    draw_cell(ax, (0, p2_starstar), (1, p2_star), info=info, label='B', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_1'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}, 'b':'x'}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-h_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': r'p_2-\left(1-p_2\right)\frac{p_2^{**}}{1-p_2^{**}}'}
    draw_cell(ax, (p1_star, p2_star), (1, 1), info=info, label='C', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': '0'}
    draw_cell(ax, (0, 0), (1, p2_starstar), info=info, label='D', params=params)

    textcolor, delta = 'black', 0.03
    ax.text(fig_width+delta, p2_star*fig_height, f'$p_2^*$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(fig_width+delta, p2_starstar*fig_height, '$p_2^{**}$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(0-delta, 0-delta, f'$0$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(p1_star*fig_width, fig_height+delta, f'$p_1^*$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.set_xlabel('$p_1$', fontsize=14)
    ax.set_ylabel('$p_2$', fontsize=14)

    #set boundaries of the figure at fig_width and fig_height
    ax.set_xlim([0, fig_width])
    ax.set_ylim([0, fig_height])
    #remove ticks and labels on axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #show the figute
    plt.savefig(f'figures/{name}.png', dpi=600)

def Mechanism_a2_l2_h1(params = params):
    name = 'Mechanism_a2_l2_h1'
    params = params.copy()
    params['p2_star'] = 0.45
    params['fig_height'] = 1.7
    params['fig_width'] = 1.8
    params['beta'] = 0.3
    params['payoff_size'] = 0.7
    params['p2_star'] = 0.48
    p2_star, p2_starstar, p1_star, fig_height, fig_width, payoff_size, beta = params['p2_star'], params['p2_starstar'], params['p1_star'], params['fig_height'], params['fig_width'], params['payoff_size'], params['beta']
    #fig_height = 1.5
    #draw a box and divide it with three horizontal lines
    fig, ax = plt.subplots(figsize=(5*fig_width, 5*fig_height))
    info = {'l1':{'l2':{'s':{'q':0.2, 'c':'y_A', 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':1},'u':{1:{'p':0.7, 'q':1, 'tq':'-h_1'}, 2:{'p':0.3, 'q':1, 'tq':'-h_1+l_2-l_1', 'c':'z_A'}}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': r'p_2-\left(1-p_2\right)\frac{p_2^*}{1-p_2^*}'}
    draw_cell(ax, (0, p2_star), (p1_star, 1), info=info, label='A', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-h_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0, 'tn':'h_2-h_1'}},
                'L':''},
            'Lambda1': 'p_1', 'Lambda2': '0'}
    draw_cell(ax, (0,0), (p1_star, p2_star), info=info, label='B', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}, 'b':'x'}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-h_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': 'p_2'}
    draw_cell(ax, (p1_star, 0), (1, 1), info=info, label='C', params=params)

    textcolor, delta = 'black', 0.03
    ax.text(0-delta, p2_star*fig_height, f'$p_2^*$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(0-delta, 0-delta, f'$0$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(p1_star*fig_width, fig_height+delta, f'$p_1^*$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.set_xlabel('$p_1$', fontsize=14)
    ax.set_ylabel('$p_2$', fontsize=14)

    #set boundaries of the figure at fig_width and fig_height
    ax.set_xlim([0, fig_width])
    ax.set_ylim([0, fig_height])
    #remove ticks and labels on axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #show the figute
    plt.savefig(f'figures/{name}.png', dpi=600)

def Mechanism_a1_l2_h1(params = params):
    name = 'Mechanism_a1_l2_h1'
    params = params.copy()
    params['p2_star'] = 0.45
    params['fig_height'] = 1.7
    params['fig_width'] = 1.8
    params['beta'] = 0.4
    params['payoff_size'] = 0.7
    params['p2_star'] = 0.48
    p2_star, p2_starstar, p1_star, fig_height, fig_width, payoff_size, beta = params['p2_star'], params['p2_starstar'], params['p1_star'], params['fig_height'], params['fig_width'], params['payoff_size'], params['beta']
    #fig_height = 1.5
    #draw a box and divide it with three horizontal lines
    fig, ax = plt.subplots(figsize=(5*fig_width, 5*fig_height))
    info = {'l1':{'l2':{'s':{'q':0.2, 'c':'y_A', 'tn':'l_1'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-h_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': r'p_2-\left(1-p_2\right)\frac{p_2^*}{1-p_2^*}'}
    draw_cell(ax, (0, p2_star), (p1_star, 1), info=info, label='A', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'l_2'},'u':{'q':0}}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-h_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}},
                'L':''},
            'Lambda1': 'p_1', 'Lambda2': 'p_2'}
    draw_cell(ax, (0,0), (1, p2_star), info=info, label='B', params=params)
    info = {'l1':{'l2':{'s':{'q':0, 'tn':'l_1'},'u':{'q':0}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}, 'b':'x'}}, 
            'h1':{'l2':{'s':{'q':1},'u':{'q':1, 'tq':'-h_1'}}, 
                'h2':{'s':{'q':0, 'tn':'h_2'},'u':{'q':0}}},
            'Lambda1': 'p_1', 'Lambda2': 'p_2'}
    draw_cell(ax, (p1_star, p2_star), (1, 1), info=info, label='C', params=params)

    textcolor, delta = 'black', 0.03
    ax.text(0-delta, p2_star*fig_height, f'$p_2^*$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(0-delta, 0-delta, f'$0$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.text(p1_star*fig_width, fig_height+delta, f'$p_1^*$', horizontalalignment='center', verticalalignment='center', color = textcolor)
    ax.set_xlabel('$p_1$', fontsize=14)
    ax.set_ylabel('$p_2$', fontsize=14)

    #set boundaries of the figure at fig_width and fig_height
    ax.set_xlim([0, fig_width])
    ax.set_ylim([0, fig_height])
    #remove ticks and labels on axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #show the figute
    plt.savefig(f'figures/{name}.png', dpi=600)

Mechanism_a1_h1_l2(params)
plt.show()
Mechanism_a1_l2_h1(params)
Mechanism_a2_l2_h1(params)
Mechanism_a2_h1_l2(params)