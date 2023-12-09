# To activate virtual environment, run .venv\Scripts\activate.ps1
# After activating virtual environment, make sure that requiremnts are installed by running: pip install -r requirements.txt
# To test run it locally, run: streamlit run BwMpayoffs.py
# To stop the local server, press Ctrl+C (in the terminal window)

import streamlit as st
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from Data import *     

from Mechanism import Mechanism
        
class Plot:

    def __init__(self, p1, p2, mechanism='a2', case = 'l2<h1', mode='deployment', sticky = True, mechanism_payoff = 'u(l)', mixing = None):
        self.mechanism = Mechanism(mechanism = mechanism, case = case, mechanism_payoff = mechanism_payoff)
        self.mixing = mixing
        self.mode = mode        

        self.p1 = p1
        if sticky and abs(p1-self.mechanism.p1star)<0.03:
            self.p1 = self.mechanism.p1star
        self.p2 = p2
        if sticky and abs(p2-self.mechanism.p2star)<0.03:
            self.p2 = self.mechanism.p2star
        self.p = [self.p1, self.p2]
        
        self.big_gap = 0.01
        self.gap = 0.01
        self.set_plot_boundaries()

            
    def set_plot_boundaries(self):
        self.L0 = self.mechanism.Ml(self.p)
        self.H0 = self.mechanism.Mh(self.p)
        if self.mechanism.player == '1':      
            self.LMax = self.L0 + Delta/(1-self.mechanism.p1star)
            self.HMax = self.H0 + Delta/self.mechanism.p1star
        if self.mechanism.player == '2':
            self.LMax = self.L0 + Delta/(1-self.mechanism.p2star)
            self.HMax = self.H0 + beta*(h2-l2)
        
    def draw_boundary(self, boundary, color = 'grey', label = '', fillcolor = None, linestyle = '-', transfer = 0):  
        for x in boundary:    
            plt.plot(x.l()-transfer, x.h()-transfer, marker='o', color=color)
        ls = [x.l()-transfer for x in boundary]
        hs = [x.h()-transfer for x in boundary]
        plt.plot(ls, hs, color=color, linestyle=linestyle)
        if fillcolor is not None:
            plt.fill(ls+[-1], hs+[-1], facecolor=fillcolor)
        if label != '':
            plt.text(boundary[0].l() - self.gap - transfer, boundary[0].h() - transfer, label, horizontalalignment='right', verticalalignment='center')
    
    def plot_belief_space(self, extras = []):
        ax = plt.gca()  # Get current axes
        square = patches.Rectangle((0, 0), 1, 1, fill=False)
        ax.add_patch(square)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for zone in self.mechanism.zone_rectangles:
            NW = self.mechanism.zone_rectangles[zone][0]
            SE = self.mechanism.zone_rectangles[zone][1]
            NE = [SE[0], NW[1]]
            SW = [NW[0], SE[1]]
            plt.plot([NW[0], NE[0]], [NW[1], NE[1]], color='grey', linestyle='-')
            plt.plot([NW[0], SW[0]], [NW[1], SW[1]], color='grey', linestyle='-')
            plt.plot([NE[0], SE[0]], [NE[1], SE[1]], color='grey', linestyle='-')
            plt.plot([SW[0], SE[0]], [SW[1], SE[1]], color='grey', linestyle='-')
            plt.text(NW[0]+0.1, NW[1]-0.1, 'Zone '+zone, color = (0.8, 0.678, 0), horizontalalignment='center', verticalalignment='center')
        plt.text(-0.05, self.p2, '$p_2$', horizontalalignment='center', verticalalignment='center')
        plt.text(self.p1, -0.05,  '$p_1$', horizontalalignment='center', verticalalignment='center')
        #plt.text(-0.05, self.p2star, '$p^{*}_2$', horizontalalignment='center', verticalalignment='center')
        #plt.text(self.p1star, -0.05,  '$p^{*}_1$', horizontalalignment='center', verticalalignment='center')
        plt.axvline(x=self.p1, color='grey', linestyle='--')
        plt.axhline(y=self.p2, color='grey', linestyle='--')
        
        if 'The Gap' in extras:
            if self.mechanism.player == '1':
                plt.axvline(x=self.p1, color='pink', linestyle='--')
            if self.mechanism.player == '2':
                plt.axhline(y=self.p2, color='pink', linestyle='--')
            #Draw the gap
            X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
            colors = [(1, 1, 1), (0.564, 0.933, 0.564)]  # White to light green
            n_bins = 100  # Increase this for smoother color transitions
            cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors, N=n_bins)
            plt.pcolormesh(X, Y, self.mechanism.shade(X, Y), cmap=cmap, shading='auto')

        if 'Equations' in extras:
            #Draw the linearly transferable zones
            zones = self.mechanism.find_linearly_transferable_zones()
            for zone in zones:
                x, y = self.mechanism.zone_rectangles[zone]
                x0, x1 = x  # Example values
                y0, y1 = y  # Example values
                rectangle = patches.Rectangle((x0, x1), y0-x0, y1-x1, fill=True, color=(0.9, 0.9, 0.9), hatch=None)
                ax.add_patch(rectangle)
                # Add text in the middle of the rectangle
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                text_str = "Linearly\ntransferable\npayoffs"  # Multi-line text
                plt.text(mid_x, mid_y, text_str, horizontalalignment='center', verticalalignment='center', fontsize='smaller')

        plt.plot(self.p1, self.p2, marker='o', color='red')
        plt.title(f"Belief space")
        plt.xlim(0, 1)  # Limit x-axis from 0 to 1
        plt.ylim(0, 1)  # Limit y-axis from 0 to 1

    def plot_feasible(self):
        plt.title(f"Payoffs in mechanism $"+self.mechanism.name()+f"$")
        # Remove tick markks and axis
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        #Draw axis lines
        L0, H0, LMax, HMax = self.L0, self.H0, self.LMax, self.HMax
        Llabel = '$l_'+self.mechanism.player+'$'
        Hlabel = '$h_'+self.mechanism.player+'$'
        title = f'Feasible and IC payoffs of player '+self.mechanism.player
        plt.plot([L0, LMax], [H0, H0], color='black', linestyle='-')
        plt.plot([L0, L0], [H0, HMax], color='black', linestyle='-')
        plt.text(LMax, H0-self.gap, Llabel, horizontalalignment='center', verticalalignment='center')
        plt.text(L0-self.gap, HMax, Hlabel, horizontalalignment='center', verticalalignment='top')
        plt.title(title)
        #Draw feasible set, the allocation, and strong Pareto frontier.
        p = [self.p1, self.p2]
        self.draw_boundary(self.mechanism.find_frontier(p), color = 'lightgrey', linestyle=':')
        self.draw_boundary(self.mechanism.find_feasible(p), color = 'black', fillcolor = 'lightgrey', linestyle='--')
        self.draw_boundary(self.mechanism.find_allocation(p), color = 'blue')
        #Limit the plot
        plt.xlim(L0 - self.big_gap, LMax + self.big_gap)  # Limit x-axis
        plt.ylim(H0 - self.big_gap, HMax + self.big_gap)  # Limit y-axis

    def plot_gap(self):
        plt.title(f"Payoffs in mechanism $"+self.mechanism.name()+f"$, zone {self.mechanism.find_zones(self.p)[0]}")
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        x0 = 0
        y0 = 0
        xMax = 1
        yMax = max(h2, h1)
        ylabel = 'payoffs'
        xlabel = '$p_'+self.mechanism.mechanism[1]+'$'
        plt.plot([x0, xMax], [y0, y0], color='pink', linestyle='-')
        plt.plot([x0, x0], [y0, yMax], color='black', linestyle='-')
        plt.text(xMax, y0 - self.gap, xlabel, horizontalalignment='center', verticalalignment='center')
        plt.text(x0 - self.gap, yMax, ylabel, horizontalalignment='center', verticalalignment='top')
        
        ps, allocations, monopolies, current = self.mechanism.find_payoff_crossections(self.p)    
        plt.fill(ps+ps[::-1], allocations+monopolies[::-1], facecolor='lightgreen')
        plt.plot(ps, monopolies, color='lightgreen', linestyle='--')
        plt.plot(ps, allocations, color='green', linestyle='-')
        plt.plot(current[0], current[1], marker='o', color='blue')
        
        plt.plot([0.6, 0.75], [yMax-0.2, yMax-0.2], color='lightgreen', linestyle='--')
        plt.text(0.8, yMax-0.3, 'monopoly payoffs')
        plt.plot([0.6, 0.75], [yMax-0.5, yMax-0.5], color='green', linestyle='-')
        plt.text(0.8, yMax-0.6, 'allocation payoffs')

    def plot_allocations(self):

        plt.title(f"Allocation in mechanism $"+self.mechanism.name()+f", zone ${self.mechanism.find_zones(self.p)[0]}$")
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        a = self.mechanism.find_allocation(self.p)[0].allocation
        x, y = 0, 0
        for type1 in ['l1', 'h1']:
            for type2 in ['l2', 'h2']:
                q = a[type1][type2]
                xp = x + 5*(1-q)
                rectangle = plt.Rectangle((xp, y), 5*q, 5, fill=True, color = 'lightgrey')
                ax.add_patch(rectangle)
                y +=5
            x, y = x+5, 0
        rectangle = plt.Rectangle((0, 0), 10, 10, fill=False)
        ax.add_patch(rectangle)
        plt.plot([5,5], [0, 10], color='black', linestyle='-')
        plt.plot([0, 10], [5, 5], color='black', linestyle='-')
        plt.text(2.5, -1, '$l_1$', horizontalalignment='center', verticalalignment='center')
        plt.text(7.5, -1, '$h_1$', horizontalalignment='center', verticalalignment='center')
        plt.text(-1, 2.5, '$l_2$', horizontalalignment='center', verticalalignment='center')
        plt.text(-1, 7.5, '$h_2$', horizontalalignment='center', verticalalignment='center')

    def plot_equations(self):
        plt.title(f"Payoff equations in mechanism $"+self.mechanism.name()+f"$, zone {self.mechanism.find_zones(self.p)[0]}")
        # Remove tick markks and axis
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        payoffs_local = self.mechanism.find_payoff_equations(self.p)
        r = 1
        for player_type in payoffs_local:
            r -= 0.1
            payoff = payoffs_local[player_type].replace("x", r"\beta ")
            if player_type[1] == self.mechanism.player:
                color = 'blue' 
            else: 
                color = 'black' 
            plt.text(0, r, f'{player_type}: '+payoff, horizontalalignment='left', verticalalignment='center', color = color)

    def plot_mixing(self):
        plt.title(f"Payoffs in mechanism $"+self.mechanism.name()+f"$")
        # Remove tick markks and axis
        ax = plt.gca()  # Get current axes
        ax.axis('off')

        #Draw axis lines
        pa1 = Mechanism(mechanism = 'a1', case = self.mechanism.case, player='1')
        pa2 = Mechanism(mechanism = 'a2', case = self.mechanism.case, player='1') 
        pa2_allocations = pa2.find_allocation(self.p)
        pa1_allocations = pa1.find_allocation(self.p)

        L0, H0, LMax, HMax = self.L0, self.H0, self.LMax, self.HMax
        Llabel = '$l_1$'
        Hlabel = '$h_1$'
        title = f'Payoffs of player 1'
        plt.plot([L0, LMax], [H0, H0], color='black', linestyle='-')
        plt.plot([L0, L0], [H0, HMax], color='black', linestyle='-')
        plt.text(LMax, H0-self.gap, Llabel, horizontalalignment='center', verticalalignment='center')
        plt.text(L0-self.gap, HMax, Hlabel, horizontalalignment='center', verticalalignment='top')
        plt.title(title)
    
        #Draw allocations 
        mechanisms = self.mixing['mechanisms']
        transfer = self.mixing['transfer'] * Delta * self.mechanism.shade1(self.p1)
        if mechanisms['a1']:
            color = 'lightblue'
            allocations = pa1_allocations
            self.draw_boundary(allocations, label = '$a_1$', color = color)
            line = [[allocations[0].l()-20, allocations[0].l()+20], [allocations[0].h()+20*(1-self.p1)/self.p1, allocations[0].h()-20*(1-self.p1)/self.p1]]
            plt.plot(line[0], line[1], color = color, linestyle = '--')
        if mechanisms['a2'] and mechanisms['a2-gap']:    
            for a in pa2_allocations:
                plt.arrow(a.l()-0.2*transfer, a.h()-0.2*transfer, -0.6*transfer, -0.6*transfer, color = 'lightgrey')
        if mechanisms['a2']:
            color = 'lightgrey'
            allocations = pa2_allocations
            self.draw_boundary(allocations, label = '$a_2$', color = color)
            line = [[allocations[0].l()-20, allocations[0].l()+20], [allocations[0].h()+20*(1-self.p1)/self.p1, allocations[0].h()-20*(1-self.p1)/self.p1]]
            plt.plot(line[0], line[1], color = color, linestyle = '--')
        if mechanisms['a2-gap']:
            color = 'brown'
            allocations = pa2_allocations
            self.draw_boundary(pa2_allocations, label = '$a_2$-Gap', color = color, transfer = transfer)
            line = [[allocations[0].l()-transfer-20, allocations[0].l()-transfer+20], [allocations[0].h()-transfer+20*(1-self.p1)/self.p1, allocations[0].h()-transfer-20*(1-self.p1)/self.p1]]
            plt.plot(line[0], line[1], color = color, linestyle = '--')
        if mechanisms['max(a1, a2-gap)'] and self.p1 != self.mechanism.p1star:
            color = 'green'
            allocations = pa2_allocations
            a2_g = [allocations[0].l()-transfer, allocations[0].h()-transfer]
            allocations = pa1_allocations
            a1 = [allocations[0].l(), allocations[0].h()]
            a = [max(a1[0], a2_g[0]), max(a1[1], a2_g[1])]
            plt.plot(a[0], a[1], marker='o', color=color)
            plt.plot([a[0], a1[0]], [a[1], a1[1]], color = color, linestyle = ':')
            plt.plot([a[0], a2_g[0]], [a[1], a2_g[1]], color = color, linestyle = ':')
            line = [[a[0]-20, a[0]+20], [a[1]+20*(1-self.p1)/self.p1, a[1]-20*(1-self.p1)/self.p1]]
            plt.plot(line[0], line[1], color = color, linestyle = '--')
        if mechanisms['max(a1, a2-gap)'] and self.p1 == self.mechanism.p1star:
            color = 'green'
            allocations = pa1_allocations
            a1 = [allocations[0].l(), allocations[0].h()]
            plt.plot(a1[0], a1[1], marker='o', color=color)
            
        #Limit the plot
        plt.xlim(L0 - self.big_gap - self.mixing['transfer'] * Delta, LMax + self.big_gap)  # Limit x-axis
        plt.ylim(H0 - self.big_gap - self.mixing['transfer'] * Delta, HMax + self.big_gap)  # Limit y-axis

    def draw(self, plots):

        self.n_plots = len(plots)
        fig = plt.figure(figsize=(6*self.n_plots,3.5))

        for k in range(self.n_plots):
            plt.subplot(1, self.n_plots, k+1)  # 1 row, 2 columns, 2nd subplot
            if plots[k] == 'Belief space':
                extras = []
                if 'The Gap' in plots: extras.append('The Gap')
                if 'Equations' in plots: extras.append('Equations')
                self.plot_belief_space(extras)
            if plots[k] == 'Feasible payoffs':
                self.plot_feasible()
            if plots[k] == 'The Gap':
                self.plot_gap()
            if plots[k] == 'Allocations':
                self.plot_allocations()
            if plots[k] == 'Equations':
                self.plot_equations()
            if plots[k] == 'Mixing and matching':
                self.plot_mixing()
                
        #Final
        if self.mode == 'development':
            plt.show()
        else:
            return fig
    

mode = 'deployment' 
#mode = 'development'

if mode == 'deployment':
    # Create columns
    col0, col1, col2, col3, col4 = st.columns([2, 5,5, 2, 2])
    with col0:
        page = st.radio("Page", ["Feasible payoffs", "The Gap", "Equations", "Allocations", "Mixing and matching"])
    # Create slider widgets in columns
    with col1:
        p1_value = st.slider('p1:', min_value=0.001, max_value=0.999, value=0.3, step=0.01, format='%.3f')
    with col2:
        p2_value = st.slider('p2:', min_value=0.001, max_value=0.999, value=0.2, step=0.01, format='%.3f')
    # Create a radio button selector in a column
    # Mechanism selection using buttons for a horizontal layout
    # Display the selected page
    if page == "Feasible payoffs":
        with col3:
            mechanism = st.radio("Mechanism", ['a1', 'a2'])
        plot = Plot(p1_value, p2_value, mechanism = mechanism, mode=mode)
        fig = plot.draw(['Belief space', 'Feasible payoffs'])
    if page == "The Gap":
        with col3:
            mechanism = st.radio("Mechanism", ['a1', 'a2'])
        with col4:
            payoffs = st.radio("Payoffs", ['expected', 'u(l)', 'u(h)'] )
        plot = Plot(p1_value, p2_value, mechanism = mechanism, mode=mode, mechanism_payoff = payoffs)
        fig = plot.draw(['Belief space', 'The Gap'])
    if page == "Equations":
        with col3:
            mechanism = st.radio("Mechanism", ['a1', 'a2'])
        plot = Plot(p1_value, p2_value, mechanism = mechanism, mode=mode)
        fig = plot.draw(['Belief space', 'Equations'])
    if page == "Allocations":
        with col3:
            mechanism = st.radio("Mechanism", ['a1', 'a2'])
        plot = Plot(p1_value, p2_value, mechanism = mechanism, mode=mode)
        fig = plot.draw(['Belief space', 'Allocations'])
    if page == "Mixing and matching":
        mechanisms = {}
        with col3:
            checkbox_status = {'a1':True, 'a2':False, 'a2-gap':True, 'max(a1, a2-gap)':False}
            for m in checkbox_status:
                mechanisms[m] = st.checkbox(m, checkbox_status[m])
        with col4:
            transfer = st.slider('gap transfered', min_value=float(0), max_value=float(1), value=float(1), step=0.01, format='%.3f')
        mixing = {'mechanisms':mechanisms, 'transfer':transfer}
        plot = Plot(p1_value, p2_value, mechanism = 'a2', mixing = mixing, mode=mode)
        fig = plot.draw(['Belief space', 'Mixing and matching'])
    st.pyplot(fig)
    
else:
    #feature = 'Mixing and matching'
    #feature = 'Allocations'
    #feature = 'Equations'
    #feature = 'The Gap'
    feature = 'Feasible payoffs'
    mixing = {'mechanisms':{'a1':True, 'a2':False, 'a2-gap':True, 'max(a1, a2-gap)':True}, 'transfer':1}
    mechanism_payoff = 'u(l)'
    plot=Plot(0.4, 0.2, mechanism='a2', mode=mode, mixing = mixing, mechanism_payoff = mechanism_payoff).draw(['Belief space', feature])
    plot = Plot(0.3, p2_star, mechanism='a1', mode='development').draw(['Belief space', feature])
    plot=Plot(0.3, 0.5, mechanism='a1', mode='development').draw(['Belief space', 'Feasible payoffs'])
    plot=Plot(0.3, 0.2, mechanism='a2', case = 'l2<h1', mode='development').draw(['Belief space', 'Feasible payoffs'])
    plot = Plot(0.3, 0.2, mechanism='a2', case = 'l2<h1', mode='development').draw(['Belief space', 'Feasible payoffs'])