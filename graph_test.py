# To activate virtual environment, run .venv\Scripts\activate.ps1
# After activating virtual environment, make sure that requiremnts are installed by running pip install -r requirements.txt
# To test run it locally, run streamlit run graph_test.py

import streamlit as st
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

l1 = 0
h1 = 2
l2 = 1
h2 = 3

beta = 0.5

p2_star = (l2-l1)/(h2-l1)
p2_star_star = (l2-l1)/(h2-l2)

p1_star = {'l2<h1':{'a1':beta/(beta + (1-beta)*(h2-h1)/(h2-l1)), 'a2':beta/(beta + (1-beta)*(h2-h1)/(h2-l2))}, \
        'l2>h1':{'a1':0, 'a2':1}}
Delta = beta *(1-beta) *(h2-h1) *(h2-l2) /(h2-l1) /(h2-beta*l2-(1-beta)*h1)

payoffs = {'l2<h1':{\
                'a1':{'A':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)+\left(1-x\right)\left(1-p_{2}\right)\frac{h_{2}-h_{1}}{h_{2}-l_{2}}\left(l_{2}-l_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$',}, \
                    'B':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)+\left(1-x\right)p_{2}\left(h_{2}-h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)-\left(1-x\right)p_{1}\left(h_{2}-h_{1}\right)+x\left(1-p_{1}\right)\left(h_{2}-l_{2}\right)$',}, \
                    'C':{'l1':r'$x\left(\left(1-p_{2}\right)l_{2}+p_{2}h_{2}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'}},

                'a2':{'A':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)+\left(1-x\right)p_{1}\frac{h_{2}-h_{1}}{h_{2}-l_{1}}\left(l_{2}-l_{1}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'}, \
                    'B':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)+x\left(1-p_{1}\right)\left(h_{2}-l_{2}\right)$'}, \
                    'C':{'l1':r'$xM_{1}\left(l_{1}\right)+x\left(1-p_{1}\right)\left(l_{2}-l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)+x\left(1-p_{1}\right)\left(l_{2}-l_{1}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'}}},

           'l2>h1':{\
                'a1':{'A':{'l1':r'', \
                        'h1':r'', \
                        'l2':r'', \
                        'h2':r''}, \
                    'B':{'l1':r'', \
                        'h1':r'', \
                        'l2':r'', \
                        'h2':r''}, \
                    'C':{'l1':r'', \
                        'h1':r'', \
                        'l2':r'', \
                        'h2':r''},\
                    'D':{'l1':r'', \
                        'h1':r'', \
                        'l2':r'', \
                        'h2':r''}},

                'a2':{'A':{'l1':r'', \
                        'h1':r'', \
                        'l2':r'', \
                        'h2':r''}, \
                    'B':{'l1':r'', \
                        'h1':r'', \
                        'l2':r'', \
                        'h2':r''}, \
                    'C':{'l1':r'', \
                        'h1':r'', \
                        'l2':r'', \
                        'h2':r''},\
                    'D':{'l1':r'', \
                        'h1':r'', \
                        'l2':r'', \
                        'h2':r''}}}}

def Ml1(p2):
    return beta*max(l2, (1-p2)*l1 + p2*h2)

def Mh1(p2):
    return beta*max(l2, (1-p2)*h1 + p2*h2)

def Ml2(p1):
    return (1-beta)*max(l2, (1-p1)*l2 + p1*h1)

def Mh2(p1):
    return (1-beta)*h2

def payoffs1(p1, p2, allocation, h_share):
    Welfare = (1-p1)*(1-p2)*(l2 + allocation['l1']['l2']*(l1-l2)) 
    Welfare += (1-p1)*p2*(h2 + allocation['l1']['h2']*(l1-h2)) 
    Welfare += p1*(1-p2)*(l2 + allocation['h1']['l2']*(h1-l2)) 
    Welfare += p1*p2*(h2 + allocation['h1']['h2']*(h1-h2))
    ql2 = (1-p1)*(1-allocation['l1']['l2']) + p1*(1-allocation['h1']['l2'])
    L2 = Ml2(p1)
    H2 = max(Mh2(p1), L2 + ql2*(h2-l2))
    Welfare1 = Welfare - (1-p2)*L2 - p2*H2
    if h_share == 0:
        ql1 = (1-p2)*allocation['l1']['l2'] + p2*allocation['l1']['h2']
        L1 = Welfare1 - p1*ql1*(h1-l1)
        H1 = L1 + ql1*(h1-l1)
    else: 
        qh1 = (1-p2)*allocation['h1']['l2'] + p2*allocation['h1']['h2']
        L1 = Welfare1 - p1 * qh1 * (h1-l1)
        H1 = L1 + qh1*(h1-l1)
        
    return [L1, H1, L2, H2, Welfare]

def payoffs2(p1, p2, allocation, h_share):
    Welfare = (1-p1)*(1-p2)*(l2 + allocation['l1']['l2']*(l1-l2)) 
    Welfare += (1-p1)*p2*(h2 + allocation['l1']['h2']*(l1-h2)) 
    Welfare += p1*(1-p2)*(l2 + allocation['h1']['l2']*(h1-l2)) 
    Welfare += p1*p2*(h2 + allocation['h1']['h2']*(h1-h2))
    ql1 = (1-p2)*allocation['l1']['l2'] + p2*allocation['l1']['h2']
    qh1 = (1-p2)*allocation['h1']['l2'] + p2*allocation['h1']['h2']
    ql2 = (1-p1)*(1-allocation['l1']['l2']) + p1*(1-allocation['h1']['l2'])
    qh2 = (1-p1)*(1-allocation['l1']['h2']) + p1*(1-allocation['h1']['h2'])
    L1 = Ml1(p2)
    H1 = Mh1(p2)
    Welfare2 = Welfare - (1-p1)*L1 - p1*H1
    if h_share == 0:
        L2 = Welfare2 - p2 * ql2 * (h2-l2)
        H2 = L2 + ql2 * (h2-l2)
    else: 
        L2 = Welfare2 - p2 * qh2 * (h2-l2)
        H2 = L2 + qh2 * (h2-l2)
        
    return [L2, H2, L1, H1, Welfare]

def line_intersection(line1, line2):
    xdiff = np.array([line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]])
    ydiff = np.array([line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('Lines do not intersect')

    d = (det(*line1), det(*line2))
    x0 = det(d, xdiff) / div
    x1 = det(d, ydiff) / div
    return [x0, x1]

def plot_mechanism(p1, p2, mechanism='a2', case = 'l2<h1', equations="No", mode='deployment'):

    p1star = p1_star[case][mechanism]
    p2star = p2_star

    if abs(p1-p1star)<0.05:
        p1 = p1star
    if abs(p2-p2star)<0.05:
        p2 = p2star
    
    if equations == 'Yes':
        n_plots = 3
    else:
        n_plots = 2
    fig = plt.figure(figsize=(6*n_plots,5))

    # Belief space
    plt.subplot(1, n_plots, 1)  # 1 row, 2 columns, 1st subplot
    ax = plt.gca()  # Get current axes
    square = patches.Rectangle((0, 0), 1, 1, fill=False)
    ax.add_patch(square)
    if mechanism == 'a2':
        plt.axvline(x=p1star, color='grey', linestyle='-')
        plt.plot([0, p1star], [p2star, p2star], color='grey', linestyle='-')
    else:
        plt.plot([p1star, p1star], [1, p2star], color='grey', linestyle='-')
        plt.plot([0, 1], [p2star, p2star], color='grey', linestyle='-')
    plt.text(-0.05, p2star, '$p^{*}_2$', horizontalalignment='center', verticalalignment='center')
    plt.axvline(x=p1, color='grey', linestyle='--')
    plt.axhline(y=p2, color='grey', linestyle='--')
    plt.plot(p1, p2, marker='o', color='red')
    plt.title(f"Belief space")
    plt.xlim(0, 1)  # Limit x-axis from 0 to 1
    plt.ylim(0, 1)  # Limit y-axis from 0 to 1

    # Payoff graph
    big_gap = 0.01
    gap = 0.01
    plt.subplot(1, n_plots, 2)  # 1 row, 2 columns, 2nd subplot
    plt.title(f"Payoffs in mechanism {mechanism}")
    # Remove tick markks and axis
    ax = plt.gca()  # Get current axes
    ax.axis('off')
    
    #Draw axis lines
    if mechanism == 'a2':
        L0 = Ml1(p2)
        H0 = Mh1(p2)
        LMax = L0 + Delta/(1-p1star)
        HMax = H0 + Delta/p1star
        Llabel = '$l_1$'
        Hlabel = '$h_1$'
        title = f"Player 1 payoffs"
    else:
        L0 = Ml2(p1)
        H0 = Mh2(p1)
        LMax = L0 + Delta/(1-p2star)
        HMax = H0 + beta*(h2-l2)
        Llabel = '$l_2$'
        Hlabel = '$h_2$'
        title = f"Player 2 payoffs"
    plt.plot([L0, LMax], [H0, H0], color='black', linestyle='-')
    plt.plot([L0, L0], [H0, HMax], color='black', linestyle='-')
    plt.text(LMax, H0-gap, Llabel, horizontalalignment='center', verticalalignment='center')
    plt.text(L0-gap, HMax, Hlabel, horizontalalignment='center', verticalalignment='top')
    plt.title(title)
    
    #Draw the Pareto frontier
    if mechanism == 'a2':
        if p2>p2star:
            share = max(beta*(1-(1-p1star)/p1star*p1/(1-p1)), 0)
        else:
            share = 0
        x = payoffs1(p1, p2, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 0)
        y = payoffs1(p1, p2, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 1)
        plt.plot([x[0], y[0]], [x[1], y[1]], color='black', linestyle='--')
        plt.plot(x[0], x[1], marker='o', color='grey')
        plt.plot(y[0], y[1], marker='o', color='grey')
        ax.fill([x[0], y[0], -1], [x[1], y[1], -1], facecolor='lightgrey')
        line=[x, y]
        lineL = [[L0,H0],[LMax,H0]]
        lineH = [[L0,H0],[L0,HMax]]
        iAB = line_intersection(line, lineH)
        iC = line_intersection(line, lineL)
        if p1<=p1star: #Zones A and B
            plt.plot(iAB[0], iAB[1], marker='o', color='blue')
            if p2>=p2star:
                zone = 'A'
            else:
                zone = 'B'
        if p1>=p1star: #Zone C
            plt.plot(iC[0], iC[1], marker='o', color='blue')
            zone = 'A'

        #frontier
        xstar = payoffs1(p1star, p2, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 0)
        ystar = payoffs1(p1star, p2, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 1)
        line=[xstar, ystar]
        iAB = line_intersection(line, lineH)
        iC = line_intersection(line, lineL)
        frontier = [[iAB[0], iC[0]], [iAB[1], iC[1]]]
        if p1==p1star:
            plt.plot(frontier[0], frontier[1], color='blue', linestyle='-')
        else:
            plt.plot(frontier[0], frontier[1], color='grey', linestyle=':')
    else:
        x = payoffs2(p1, p2, {'l1':{'l2':1, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 0)
        y = payoffs2(p1, p2, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 0)
        z = payoffs2(p1, p2, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 1)
        ax.fill([x[0], y[0], z[0], -1], [x[1], y[1], z[1], -1], facecolor='lightgrey')
        plt.plot([x[0], y[0], z[0]], [x[1], y[1], z[1]], color='black', linestyle='--')
        plt.plot(x[0], x[1], marker='o', color='grey')
        plt.plot(y[0], y[1], marker='o', color='grey')
        plt.plot(z[0], z[1], marker='o', color='grey')
        

        lineXY=[x, y]
        lineYZ=[y, z]
        lineL = [[L0,H0],[LMax,H0]]
        lineH = [[L0,H0],[L0,HMax]]
        iA = line_intersection(lineL, lineXY)
        iB = line_intersection(lineH, lineYZ)
        iC = line_intersection(lineL, lineYZ)
        if p2>=p2star and p1<=p1star: #Zone A
            plt.plot(iA[0], iA[1], marker='o', color='blue')
            zone = 'A'
        if p2>=p2star and p1>=p1star: #Zone C
            plt.plot(iC[0], iC[1], marker='o', color='blue')
            zone = 'C'
        if p2<=p2star: #Zone B
            plt.plot(iB[0], iB[1], marker='o', color='blue')
            zone = 'B'
        
        #frontier
        xstar = payoffs2(p1, p2star, {'l1':{'l2':1, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 0)
        ystar = payoffs2(p1, p2star, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 0)
        zstar = payoffs2(p1, p2star, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}}, 1)
        lineXY=[xstar, ystar]
        lineYZ=[ystar, zstar]
        iA = line_intersection(lineL, lineXY)
        iB = line_intersection(lineH, lineYZ)
        iC = line_intersection(lineL, lineYZ)
        if p1<=p1star: #Zone A adn B:
            frontier = [iA[0], ystar[0], iB[0]], [iA[1], ystar[1], iB[1]]
        if p1>=p1star: #Zone C and B:
            frontier = [iC[0], iB[0]], [iC[1], iB[1]]
        if p2==p2star:
            plt.plot(frontier[0], frontier[1], color='blue', linestyle='-')
        else:
            plt.plot(frontier[0], frontier[1], color='grey', linestyle=':')

    #Limit the plot
    plt.xlim(L0-big_gap, LMax+big_gap)  # Limit x-axis
    plt.ylim(H0-big_gap, HMax+big_gap)  # Limit y-axis
    
    # Equations
    if equations == "Yes":
        plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd subplot
        plt.title(f"Payoff equations in mechanism {mechanism}, zone {zone}")
        # Remove tick markks and axis
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        payoffs_local=payoffs[case][mechanism][zone]
        r = 1
        for player_type in payoffs_local:
            r -= 0.1
            payoff = payoffs_local[player_type].replace("x", r"\beta ")
            if ('1' in player_type and '2' in mechanism) or ('2' in player_type and '1' in mechanism):
                color = 'blue'
            else:
                color = 'black' 
            plt.text(0, r, f'{player_type}: '+payoff, horizontalalignment='left', verticalalignment='center', color = color)

    #Final
    if mode == 'development':
        plt.show()
    else:
        return fig

mode = 'deployment' 
#mode = 'development'

if mode == 'deployment':
    # Title
    st.title('Payoffs in robust mechanisms ')
    # Create columns
    col1, col2, col3, col4 = st.columns([5,5, 1, 1])
    # Create slider widgets in columns
    with col1:
        p1_value = st.slider('p1:', min_value=0.001, max_value=0.999, value=0.3, step=0.01, format='%.3f')
    with col2:
        p2_value = st.slider('p2:', min_value=0.0, max_value=1.0, value=0.2, step=0.01, format='%.3f')
    # Create a radio button selector in a column
    # Mechanism selection using buttons for a horizontal layout

    with col3:
        mechanism = st.radio("Mechanism", ['a1', 'a2'])

    with col4:
        equations = st.radio("Equations", ['Yes', 'No'])
    
    # Display the plot
    fig = plot_mechanism(p1_value, p2_value, mechanism = mechanism, equations=equations)
    st.pyplot(fig)

else:
    plot_mechanism(0.3, 0.2, mechanism='a1', equations = 'Yes', mode='development')
    plot_mechanism(0.3, p2_star, mechanism='a1', mode='development')
    plot_mechanism(0.3, 0.5, mechanism='a1', mode='development')
    plot_mechanism(0.3, 0.2, mechanism='a2', case = 'l2<h1', mode='development')
    plot_mechanism(0.3, 0.2, mechanism='a2', case = 'l2<h1', mode='development')