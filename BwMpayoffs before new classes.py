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

def Ml1(p2):
    return beta*max(l2, (1-p2)*l1 + p2*h2)

def Mh1(p2):
    return beta*max(l2, (1-p2)*h1 + p2*h2)

def Ml2(p1):
    return (1-beta)*max(l2, (1-p1)*l2 + p1*h1)

def Mh2(p1):
    return (1-beta)*h2

def payoffs1(p1, p2, allocation):
    h_share = allocation['h_share']
    Welfare = (1-p1)*(1-p2)*(l2 + allocation['l1']['l2']*(l1-l2)) 
    Welfare += (1-p1)*p2*(h2 + allocation['l1']['h2']*(l1-h2)) 
    Welfare += p1*(1-p2)*(l2 + allocation['h1']['l2']*(h1-l2)) 
    Welfare += p1*p2*(h2 + allocation['h1']['h2']*(h1-h2))
    ql2 = (1-p1)*(1-allocation['l1']['l2']) + p1*(1-allocation['h1']['l2'])
    L2 = Ml2(p1)
    H2 = max(Mh2(p1), L2 + ql2*(h2-l2))
    Welfare1 = Welfare - (1-p2)*L2 - p2*H2
    ql1 = (1-p2)*allocation['l1']['l2'] + p2*allocation['l1']['h2']
    qh1 = (1-p2)*allocation['h1']['l2'] + p2*allocation['h1']['h2']
    q1 = ((1-h_share)*ql1 + h_share*qh1)
    L1 = Welfare1 - p1*q1*(h1-l1)
    H1 = L1 + q1*(h1-l1)
        
    return {'l1':L1, 'h1':H1, 'l2':L2, 'h2':H2, 'Welfare':Welfare}

def payoffs2(p1, p2, allocation):
    h_share = allocation['h_share']
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
    q2 = ((1-h_share)*ql2 + h_share*qh2)
    L2 = Welfare2 - p2 * q2 * (h2-l2)
    H2 = L2 + q2 * (h2-l2)
        
    return {'l1':L1, 'h1':H1, 'l2':L2, 'h2':H2, 'Welfare':Welfare}

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

class Allocation:

    def __init__(self, p1, p2, allocation, mechanism='a2'):

        self.p1 = p1
        self.p2 = p2
        self.allocation = allocation
        self.h_share = allocation['h_share']
        self.mechanism = mechanism
        if mechanism == 'a2':
            self.player = '1'
            self.payoff_function = payoffs1
        else:
            self.player = '2'
            self.payoff_function = payoffs2
        self.payoffs = self.payoff_function(p1, p2, allocation)

    def set_player(self, player):
        self.player = player
    
    def l(self):
        return self.payoffs['l1'] if self.player == '1' else self.payoffs['l2']
    
    def h(self):
        return self.payoffs['h1'] if self.player == '1' else self.payoffs['h2']
    
    def draw(self, color = 'grey'):
        plt.plot(self.l(), self.h(), marker='o', color=color)

    def intersection(self, other, axis, value):
        if axis == 'H':
            alpha = (value - other.h())/(self.h() - other.h())
        else:
            alpha = (value - other.l())/(self.l() - other.l())
        
        new = Allocation(self.p1, self.p2, {'l1':{'l2':alpha*self.allocation['l1']['l2'] + (1-alpha)*other.allocation['l1']['l2'], \
                                                    'h2':alpha*self.allocation['l1']['h2'] + (1-alpha)*other.allocation['l1']['h2']}, \
                                                'h1':{'l2':alpha*self.allocation['h1']['l2'] + (1-alpha)*other.allocation['h1']['l2'], \
                                                    'h2':alpha*self.allocation['h1']['h2'] + (1-alpha)*other.allocation['h1']['h2']}, \
                                                'h_share':alpha*self.allocation['h_share'] + (1-alpha)*other.allocation['h_share']}, self.mechanism)
        return new
    
    def __repr__(self):
        return f"Alloc.({self.l()}, {self.h()})"

class Plot:

    def __init__(self, p1, p2, mechanism='a2', case = 'l2<h1', mode='deployment', sticky = True, mechanism_payoff = 'u(l)', mixing = None):

        self.big_gap = 0.01
        self.gap = 0.01
        self.mixing = mixing
        self.mechanism = mechanism
        self.case = case
        self.mode = mode
        self.p1star = p1_star[case][mechanism]
        self.p2star = p2_star
        if case == 'l2<h1':
            self.p2starstar = 0
        else:
            self.p2starstar = 0
        self.mechanism_payoff = mechanism_payoff
        
        self.p1 = p1
        if sticky and abs(p1-self.p1star)<0.05:
            self.p1 = self.p1star
        self.p2 = p2
        if sticky and abs(p2-self.p2star)<0.05:
            self.p2 = self.p2star

        self.n_plots = 2
        self.find_zones()

    def payoffs(self, l, h):
        if self.mechanism == 'a1':
            p = self.p2
        if self.mechanism == 'a2':
            p = self.p1
        if self.mechanism_payoff == 'expected':
            return (1-p)*l + p*h
        if self.mechanism_payoff == 'u(l)': 
            return l
        if self.mechanism_payoff == 'u(h)': 
            return h        


    def find_zones(self):
        self.zones = []
        self.zone_rectangles = {}

        if self.mechanism == 'a1' and self.case == 'l2<h1':
            self.zone_rectangles['A'] = [[0,1], [self.p1star, self.p2star]]
            self.zone_rectangles['B'] = [[0, self.p2star], [1, 0]]
            self.zone_rectangles['C'] = [[self.p1star, 1], [1, self.p2star]]
            
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            self.zone_rectangles['A'] = [[0, 1], [self.p1star, self.p2star]]
            self.zone_rectangles['B'] = [[0, self.p2star], [self.p1star, 0]]
            self.zone_rectangles['C'] = [[self.p1star, 1], [1, 0]]
            
        for zone in self.zone_rectangles:
            x = self.zone_rectangles[zone][0]
            y = self.zone_rectangles[zone][1]
            if self.p1 >= x[0] and self.p1 <=y[0] and self.p2 <= x[1] and self.p2 >=y[1]:
                self.zones.append(zone)

    def set_plot_boundaries(self, player = None):
        if player == None:
            if self.mechanism == 'a2':
                player = '1'
            else:
                player = '2'
        self.player = player
        if player == '1':
            self.L0 = Ml1(self.p2)
            self.H0 = Mh1(self.p2)
            self.LMax = self.L0 + Delta/(1-self.p1star)
            self.HMax = self.H0 + Delta/self.p1star
        if player == '2':
            self.L0 = Ml2(self.p1)
            self.H0 = Mh2(self.p1)
            self.LMax = self.L0 + Delta/(1-self.p2star)
            self.HMax = self.H0 + beta*(h2-l2)
        
    
    def find_feasible(self):
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            if self.p2>self.p2star:
                share = max(beta*(1-(1-self.p1star)/self.p1star*self.p1/(1-self.p1)), 0)
            else:
                share = 0
            x = Allocation(self.p1, self.p2, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            y = Allocation(self.p1, self.p2, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':1}, self.mechanism)
            feasible = [x, y]
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            x = Allocation(self.p1, self.p2, {'l1':{'l2':1, 'h2':0}, 'h1':{'l2': 1, 'h2':0}, 'h_share':0}, self.mechanism)
            y = Allocation(self.p1, self.p2, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            z = Allocation(self.p1, self.p2, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':1}, self.mechanism)
            feasible = [x, y, z]
        return feasible
            
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
        

    def find_frontier(self):
        self.set_plot_boundaries
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            if self.p2>self.p2star:
                share = max(beta*(1-(1-self.p1star)/self.p1star*self.p1/(1-self.p1)), 0)
            else:
                share = 0
            xstar = Allocation(self.p1star, self.p2, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            ystar = Allocation(self.p1star, self.p2, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':1}, self.mechanism)
            iAB = xstar.intersection(ystar, 'L', self.L0)
            iC = xstar.intersection(ystar, 'H', self.H0)
            frontier = [iAB, iC]
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            xstar = Allocation(self.p1, self.p2star, {'l1':{'l2':1, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            ystar = Allocation(self.p1, self.p2star, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            zstar = Allocation(self.p1, self.p2star, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':1}, self.mechanism)
            iA = xstar.intersection(ystar, 'H', self.H0)
            iB = ystar.intersection(zstar, 'L', self.L0)
            iC = ystar.intersection(zstar, 'H', self.H0)
            if self.p1<=self.p1star: #Zone A adn B:
                frontier = [iA, ystar, iB]
            if self.p1>=self.p1star: #Zone C and B:
                frontier = [iC, iB]
        return frontier

    def find_allocation(self, player = None):
        if player == None:
            player = self.mechanism[1]
        self.set_plot_boundaries()
        feasible = self.find_feasible()
        frontier = self.find_frontier()
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            if self.p1 == self.p1star:
                allocation = frontier
            else: 
                x = feasible[0]
                y = feasible[1]
                iAB = x.intersection(y, 'L', self.L0)
                iC = x.intersection(y, 'H', self.H0)
                allocation = []
                if ('A' in self.zones or 'B' in self.zones): 
                    allocation.append(iAB)
                if 'C' in self.zones: #Zone C
                    allocation.append(iC)
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            if self.p2 == self.p2star:
                allocation = frontier
            else:
                x = feasible[0]
                y = feasible[1]
                z = feasible[2]
                iA = x.intersection(y, 'H', self.H0)
                iB = y.intersection(z, 'L', self.L0)
                iC = y.intersection(z, 'H', self.H0)
                if 'A' in self.zones:
                    allocation = [iA]
                if 'B' in self.zones:
                    allocation = [iB]
                if 'C' in self.zones:
                    allocation = [iC]
        for a in allocation:
            a.set_player(player)
        return allocation
            
    def shade(self, x, y):
        # Example function, replace with your actual function
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            shade1 = np.minimum((1-x)/(1-self.p1star), x/self.p1star) 
            shade2=  np.minimum((1-y)/(1-self.p2star), (y-self.p2starstar)/(self.p2star - self.p2starstar))
            return  shade1 * shade2
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            shade1 = np.minimum((1-x)/(1-self.p1star), x/self.p1star) 
            shade2=  np.minimum((1-y)/(1-self.p2star), (y-self.p2starstar)/(self.p2star - self.p2starstar))
            return  shade1 * shade2

    
    def plot_belief_space(self, extras = []):
        ax = plt.gca()  # Get current axes
        square = patches.Rectangle((0, 0), 1, 1, fill=False)
        ax.add_patch(square)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for zone in self.zone_rectangles:
            NW = self.zone_rectangles[zone][0]
            plt.text(NW[0]+0.1, NW[1]-0.1, 'Zone '+zone, color = (0.8, 0.678, 0), horizontalalignment='center', verticalalignment='center')
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            plt.axvline(x=self.p1star, color='grey', linestyle='-')
            plt.plot([0, self.p1star], [self.p2star, self.p2star], color='grey', linestyle='-')
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            plt.plot([self.p1star, self.p1star], [1, self.p2star], color='grey', linestyle='-')
            plt.plot([0, 1], [self.p2star, self.p2star], color='grey', linestyle='-')
        plt.text(-0.05, self.p2, '$p_2$', horizontalalignment='center', verticalalignment='center')
        plt.text(self.p1, -0.05,  '$p_1$', horizontalalignment='center', verticalalignment='center')
        #plt.text(-0.05, self.p2star, '$p^{*}_2$', horizontalalignment='center', verticalalignment='center')
        #plt.text(self.p1star, -0.05,  '$p^{*}_1$', horizontalalignment='center', verticalalignment='center')
        plt.axvline(x=self.p1, color='grey', linestyle='--')
        plt.axhline(y=self.p2, color='grey', linestyle='--')
        if 'The Gap' in extras:
            if self.mechanism == 'a2':
                plt.axvline(x=self.p1, color='pink', linestyle='--')
            if self.mechanism == 'a1':
                plt.axhline(y=self.p2, color='pink', linestyle='--')
        plt.plot(self.p1, self.p2, marker='o', color='red')

        if 'The Gap' in extras and self.mechanism == 'a2' and self.case == 'l2<h1':
            #Draw the gap
            # Create a meshgrid
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(x, y)
            # Evaluate the function
            Z = self.shade(X, Y)
            # Create a custom color map from white to light green
            colors = [(1, 1, 1), (0.564, 0.933, 0.564)]  # White to light green
            n_bins = 100  # Increase this for smoother color transitions
            cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors, N=n_bins)
            # Plot the results
            plt.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')

        if 'Equations' in extras:
            #Draw the linearly transferable zones
            if self.mechanism == 'a1':
                x, y = self.zone_rectangles['A']
            if self.mechanism == 'a2':
                x, y = self.zone_rectangles['C']
            x0, x1 = x  # Example values
            y0, y1 = y  # Example values
            rectangle = patches.Rectangle((x0, x1), y0-x0, y1-x1, fill=True, color=(0.9, 0.9, 0.9), hatch=None)
            ax.add_patch(rectangle)
            # Add text in the middle of the rectangle
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            text_str = "Linearly\ntransferable\npayoffs"  # Multi-line text
            plt.text(mid_x, mid_y, text_str, horizontalalignment='center', verticalalignment='center', fontsize='smaller')

        
        plt.title(f"Belief space")
        plt.xlim(0, 1)  # Limit x-axis from 0 to 1
        plt.ylim(0, 1)  # Limit y-axis from 0 to 1

    def plot_feasible(self):
        plt.title(f"Payoffs in mechanism $a_"+self.mechanism[1]+f"$")
        # Remove tick markks and axis
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        #Draw axis lines
        self.set_plot_boundaries()
        if self.mechanism == 'a2':
            Llabel = '$l_1$'
            Hlabel = '$h_1$'
            title = f"Feasible and IC payoffs of player 1"
        else:
            Llabel = '$l_2$'
            Hlabel = '$h_2$'
            title = f"Feasible and IC payoffs of player 2"
        plt.plot([self.L0, self.LMax], [self.H0, self.H0], color='black', linestyle='-')
        plt.plot([self.L0, self.L0], [self.H0, self.HMax], color='black', linestyle='-')
        plt.text(self.LMax, self.H0-self.gap, Llabel, horizontalalignment='center', verticalalignment='center')
        plt.text(self.L0-self.gap, self.HMax, Hlabel, horizontalalignment='center', verticalalignment='top')
        plt.title(title)
        #Draw feasible set, the allocation, and strong Pareto frontier.
        self.draw_boundary(self.find_frontier(), color = 'lightgrey', linestyle=':')
        self.draw_boundary(self.find_feasible(), color = 'black', fillcolor = 'lightgrey', linestyle='--')
        self.draw_boundary(self.find_allocation(), color = 'blue')
        #Limit the plot
        plt.xlim(self.L0 - self.big_gap, self.LMax + self.big_gap)  # Limit x-axis
        plt.ylim(self.H0 - self.big_gap, self.HMax + self.big_gap)  # Limit y-axis

    def plot_mechanism_payoffs(self):
        plt.title(f"Payoffs in mechanism $a_"+self.mechanism[1]+f"$, zone {self.zones[0]}")
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        x0 = 0
        y0 = 0
        xMax = 1
        yMax = max(h2, h1)
        ylabel = 'payoffs'
        if self.mechanism == 'a2':
            xlabel = '$p_2$'
        else:
            xlabel = '$p_1$'
        plt.plot([x0, xMax], [y0, y0], color='pink', linestyle='-')
        plt.plot([x0, x0], [y0, yMax], color='black', linestyle='-')
        plt.text(xMax, y0 - self.gap, xlabel, horizontalalignment='center', verticalalignment='center')
        plt.text(x0 - self.gap, yMax, ylabel, horizontalalignment='center', verticalalignment='top')
        
        epsilon = 0.01
        allocation_payoffs = []
        monopoly_payoffs = []
        allocation_current = self.find_allocation()[0]
        allocation_current_payoffs = []
        if self.mechanism == 'a2':
            ps = [0.01, self.p2star-epsilon, self.p2star+epsilon, 0.99]
            for p2 in ps:
                p = Plot(self.p1, p2, mechanism=self.mechanism, case = self.case, sticky = False, mechanism_payoff = self.mechanism_payoff)
                allocation = p.find_allocation()[0]
                allocation_payoffs.append(p.payoffs(allocation.l(), allocation.h()))
                monopoly_payoffs.append(p.payoffs(Ml1(p2), Mh1(p2)))
            allocation_current_payoffs += [self.p2, self.payoffs(allocation_current.l(), allocation_current.h())]
        if self.mechanism == 'a1':
            ps = [0.01, self.p1star-epsilon, self.p1star+epsilon, 0.99]
            for p1 in ps:
                p = Plot(p1, self.p2, mechanism=self.mechanism, case = self.case, sticky = False, mechanism_payoff = self.mechanism_payoff)
                allocation = p.find_allocation()[0]
                allocation_payoffs.append(p.payoffs(allocation.l(), allocation.h()))
                monopoly_payoffs.append(p.payoffs(Ml2(p1), Mh2(p1)))
            allocation_current_payoffs += [self.p1, self.payoffs(allocation_current.l(), allocation_current.h())]
        
        plt.fill(ps+ps[::-1], allocation_payoffs+monopoly_payoffs[::-1], facecolor='lightgreen')
        plt.plot(ps, monopoly_payoffs, color='lightgreen', linestyle='--')
        plt.plot(ps, allocation_payoffs, color='green', linestyle='-')

        plt.plot(allocation_current_payoffs[0], allocation_current_payoffs[1], marker='o', color='blue')
        
        plt.plot([0.6, 0.75], [yMax-0.2, yMax-0.2], color='lightgreen', linestyle='--')
        plt.text(0.8, yMax-0.3, 'monopoly payoffs')
        plt.plot([0.6, 0.75], [yMax-0.5, yMax-0.5], color='green', linestyle='-')
        plt.text(0.8, yMax-0.6, 'allocation payoffs')

    def plot_allocations(self):

        plt.title(f"Allocation in mechanism $a_"+self.mechanism[1]+f'$, zone ${self.zones[0]}$')
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        a = self.find_allocation()[0].allocation
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
        plt.title(f"Payoff equations in mechanism $a_"+self.mechanism[1]+f"$, zone {self.zones[0]}")
        # Remove tick markks and axis
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        payoffs_local=payoff_equations[self.case][self.mechanism][self.zones[0]]
        r = 1
        for player_type in payoffs_local:
            r -= 0.1
            payoff = payoffs_local[player_type].replace("x", r"\beta ")
            if ('1' in player_type and '2' in self.mechanism) or ('2' in player_type and '1' in self.mechanism):
                color = 'blue'
            else:
                color = 'black' 
            plt.text(0, r, f'{player_type}: '+payoff, horizontalalignment='left', verticalalignment='center', color = color)

    def plot_mixing(self):
        plt.title(f"Payoffs in mechanism $a_"+self.mechanism[1]+f"$")
        # Remove tick markks and axis
        ax = plt.gca()  # Get current axes
        ax.axis('off')
        #Draw axis lines
        self.set_plot_boundaries()
        if self.mechanism == 'a2':
            Llabel = '$l_1$'
            Hlabel = '$h_1$'
            title = f"Feasible and IC payoffs of player 1"
        else:
            Llabel = '$l_2$'
            Hlabel = '$h_2$'
            title = f"Feasible and IC payoffs of player 2"
        plt.plot([self.L0, self.LMax], [self.H0, self.H0], color='black', linestyle='-')
        plt.plot([self.L0, self.L0], [self.H0, self.HMax], color='black', linestyle='-')
        plt.text(self.LMax, self.H0-self.gap, Llabel, horizontalalignment='center', verticalalignment='center')
        plt.text(self.L0-self.gap, self.HMax, Hlabel, horizontalalignment='center', verticalalignment='top')
        plt.title(title)

        pa1 = Plot(self.p1, self.p2, mechanism = 'a1', case = self.case, mode=self.mode)
        pa2 = Plot(self.p1, self.p2, mechanism = 'a2', case = self.case, mode=self.mode)
        
        pa2_allocations = pa2.find_allocation(player = '1')
        #Draw allocations 
        mechanisms = self.mixing['mechanisms']
        transfer = self.mixing['transfer'] * Delta * min(self.p1/self.p1star, (1-self.p1)/(1-self.p1star))
        if mechanisms['a1']:
            color = 'lightblue'
            allocations = pa1.find_allocation(player = '1')
            self.draw_boundary(allocations, label = '$a_1$', color = color)
            line = [[allocations[0].l()-20, allocations[0].l()+20], [allocations[0].h()+20*(1-self.p1)/self.p1, allocations[0].h()-20*(1-self.p1)/self.p1]]
            plt.plot(line[0], line[1], color = color, linestyle = '--')
        if mechanisms['a2'] and mechanisms['a2-gap']:    
            for a in pa2_allocations:
                plt.arrow(a.l()-0.2*transfer, a.h()-0.2*transfer, -0.6*transfer, -0.6*transfer, color = 'lightgrey')
        if mechanisms['a2']:
            color = 'lightgrey'
            allocations = pa2.find_allocation(player = '1')
            self.draw_boundary(allocations, label = '$a_2$', color = color)
            line = [[allocations[0].l()-20, allocations[0].l()+20], [allocations[0].h()+20*(1-self.p1)/self.p1, allocations[0].h()-20*(1-self.p1)/self.p1]]
            plt.plot(line[0], line[1], color = color, linestyle = '--')
        if mechanisms['a2-gap']:
            color = 'brown'
            allocations = pa2.find_allocation(player = '1')
            self.draw_boundary(allocations, label = '$a_2$-Gap', color = color, transfer = transfer)
            line = [[allocations[0].l()-transfer-20, allocations[0].l()-transfer+20], [allocations[0].h()-transfer+20*(1-self.p1)/self.p1, allocations[0].h()-transfer-20*(1-self.p1)/self.p1]]
            plt.plot(line[0], line[1], color = color, linestyle = '--')
        if mechanisms['max(a1, a2-gap)'] and self.p1 != self.p1star:
            color = 'green'
            allocations = pa2.find_allocation(player = '1')
            a2_g = [allocations[0].l()-transfer, allocations[0].h()-transfer]
            allocations = pa1.find_allocation(player = '1')
            a1 = [allocations[0].l(), allocations[0].h()]
            a = [max(a1[0], a2_g[0]), max(a1[1], a2_g[1])]
            plt.plot(a[0], a[1], marker='o', color=color)
            plt.plot([a[0], a1[0]], [a[1], a1[1]], color = color, linestyle = ':')
            plt.plot([a[0], a2_g[0]], [a[1], a2_g[1]], color = color, linestyle = ':')
            line = [[a[0]-20, a[0]+20], [a[1]+20*(1-self.p1)/self.p1, a[1]-20*(1-self.p1)/self.p1]]
            plt.plot(line[0], line[1], color = color, linestyle = '--')
        if mechanisms['max(a1, a2-gap)'] and self.p1 == self.p1star:
            color = 'green'
            allocations = pa1.find_allocation(player = '1')
            a1 = [allocations[0].l(), allocations[0].h()]
            plt.plot(a1[0], a1[1], marker='o', color=color)
            
        #Limit the plot
        plt.xlim(self.L0 - self.big_gap - self.mixing['transfer'] * Delta, self.LMax + self.big_gap)  # Limit x-axis
        plt.ylim(self.H0 - self.big_gap - self.mixing['transfer'] * Delta, self.HMax + self.big_gap)  # Limit y-axis


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
                self.plot_mechanism_payoffs()
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
    feature = 'Mixing and matching'
    #feature = 'Allocations'
    #feature = 'Equations'
    #feature = 'The Gap'
    #feature = 'Feasible payoffs'
    plot=Plot(0.4, 0.2, mechanism='a2', mode=mode, transfer = 1).draw(['Belief space', feature])
    plot = Plot(0.3, p2_star, mechanism='a1', mode='development').draw(['Belief space', feature])
    plot=Plot(0.3, 0.5, mechanism='a1', mode='development').draw(['Belief space', 'Feasible payoffs'])
    plot=Plot(0.3, 0.2, mechanism='a2', case = 'l2<h1', mode='development').draw(['Belief space', 'Feasible payoffs'])
    plot = Plot(0.3, 0.2, mechanism='a2', case = 'l2<h1', mode='development').draw(['Belief space', 'Feasible payoffs'])