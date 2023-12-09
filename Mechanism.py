# To activate virtual environment, run .venv\Scripts\activate.ps1
# After activating virtual environment, make sure that requiremnts are installed by running: pip install -r requirements.txt
# To test run it locally, run: streamlit run BwMpayoffs.py
# To stop the local server, press Ctrl+C (in the terminal window)

from Data import *

import numpy as np

def Ml1(p):
    p1, p2 = p
    return beta*max(l2, (1-p2)*l1 + p2*h2)

def Mh1(p):
    p1, p2 = p
    return beta*max(l2, (1-p2)*h1 + p2*h2)

def Ml2(p):
    p1, p2 = p
    return (1-beta)*max(l2, (1-p1)*l2 + p1*h1)

def Mh2(p):
    p1, p2 = p
    return (1-beta)*h2

def payoffs1(p, allocation):
    p1, p2 = p
    h_share = allocation['h_share']
    Welfare = (1-p1)*(1-p2)*(l2 + allocation['l1']['l2']*(l1-l2)) 
    Welfare += (1-p1)*p2*(h2 + allocation['l1']['h2']*(l1-h2)) 
    Welfare += p1*(1-p2)*(l2 + allocation['h1']['l2']*(h1-l2)) 
    Welfare += p1*p2*(h2 + allocation['h1']['h2']*(h1-h2))
    ql2 = (1-p1)*(1-allocation['l1']['l2']) + p1*(1-allocation['h1']['l2'])
    L2 = Ml2(p)
    H2 = max(Mh2(p), L2 + ql2*(h2-l2))
    Welfare1 = Welfare - (1-p2)*L2 - p2*H2
    ql1 = (1-p2)*allocation['l1']['l2'] + p2*allocation['l1']['h2']
    qh1 = (1-p2)*allocation['h1']['l2'] + p2*allocation['h1']['h2']
    q1 = ((1-h_share)*ql1 + h_share*qh1)
    L1 = Welfare1 - p1*q1*(h1-l1)
    H1 = L1 + q1*(h1-l1)
        
    return {'l1':L1, 'h1':H1, 'l2':L2, 'h2':H2, 'Welfare':Welfare}

def payoffs2(p, allocation):
    p1, p2 = p
    h_share = allocation['h_share']
    Welfare = (1-p1)*(1-p2)*(l2 + allocation['l1']['l2']*(l1-l2)) 
    Welfare += (1-p1)*p2*(h2 + allocation['l1']['h2']*(l1-h2)) 
    Welfare += p1*(1-p2)*(l2 + allocation['h1']['l2']*(h1-l2)) 
    Welfare += p1*p2*(h2 + allocation['h1']['h2']*(h1-h2))
    ql1 = (1-p2)*allocation['l1']['l2'] + p2*allocation['l1']['h2']
    qh1 = (1-p2)*allocation['h1']['l2'] + p2*allocation['h1']['h2']
    ql2 = (1-p1)*(1-allocation['l1']['l2']) + p1*(1-allocation['h1']['l2'])
    qh2 = (1-p1)*(1-allocation['l1']['h2']) + p1*(1-allocation['h1']['h2'])
    L1 = Ml1(p)
    H1 = Mh1(p)
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

    def __init__(self, p, allocation, mechanism='a2'):

        self.p1, self.p2 = p
        self.allocation = allocation
        self.h_share = allocation['h_share']
        self.mechanism = mechanism
        if mechanism == 'a2':
            self.player = '1'
            self.payoff_function = payoffs1
        else:
            self.player = '2'
            self.payoff_function = payoffs2
            self.Ml = Ml2
            self.Mh = Mh2
        self.payoffs = self.payoff_function(p, allocation)

    def set_player(self, player):
        self.player = player
    
    def l(self):
        return self.payoffs['l1'] if self.player == '1' else self.payoffs['l2']
    
    def h(self):
        return self.payoffs['h1'] if self.player == '1' else self.payoffs['h2']

    def intersection(self, other, axis, value):
        if axis == 'H':
            alpha = (value - other.h())/(self.h() - other.h())
        else:
            alpha = (value - other.l())/(self.l() - other.l())
        
        new = Allocation([self.p1, self.p2], {'l1':{'l2':alpha*self.allocation['l1']['l2'] + (1-alpha)*other.allocation['l1']['l2'], \
                                                    'h2':alpha*self.allocation['l1']['h2'] + (1-alpha)*other.allocation['l1']['h2']}, \
                                                'h1':{'l2':alpha*self.allocation['h1']['l2'] + (1-alpha)*other.allocation['h1']['l2'], \
                                                    'h2':alpha*self.allocation['h1']['h2'] + (1-alpha)*other.allocation['h1']['h2']}, \
                                                'h_share':alpha*self.allocation['h_share'] + (1-alpha)*other.allocation['h_share']}, self.mechanism)
        return new
    
    def __repr__(self):
        return f"Alloc.({self.l()}, {self.h()})"

class Mechanism:

    def __init__(self, mechanism='a2', case = 'l2<h1', mechanism_payoff = 'u(l)', player = None):
        self.mechanism = mechanism
        self.case = case
        self.p1star = p1_star[case][mechanism]
        self.p2star = p2_star
        self.mechanism_payoff = mechanism_payoff
        if case == 'l2<h1':
            self.p2starstar = 0
        else:
            self.p2starstar = 0
        if player == None:
            if self.mechanism == 'a2':
                self.player = '1'
            else:
                self.player = '2'
        else:
            self.player = player
        if self.player == '1':
            self.Ml = Ml1
            self.Mh = Mh1
        else:
            self.Ml = Ml2
            self.Mh = Mh2
        self.set_zones()

    def set_zones(self):
        self.zone_rectangles = {}
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            self.zone_rectangles['A'] = [[0,1], [self.p1star, self.p2star]]
            self.zone_rectangles['B'] = [[0, self.p2star], [1, 0]]
            self.zone_rectangles['C'] = [[self.p1star, 1], [1, self.p2star]]
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            self.zone_rectangles['A'] = [[0, 1], [self.p1star, self.p2star]]
            self.zone_rectangles['B'] = [[0, self.p2star], [self.p1star, 0]]
            self.zone_rectangles['C'] = [[self.p1star, 1], [1, 0]]
    
    def find_zones(self, p):
        p1, p2 = p
        zones = []
        for zone in self.zone_rectangles:
            x = self.zone_rectangles[zone][0]
            y = self.zone_rectangles[zone][1]
            if p1 >= x[0] and p1 <=y[0] and p2 <= x[1] and p2 >=y[1]:
                zones.append(zone)
        return zones

    def find_feasible(self, p):
        #Finds allocations on the frontier of the feasible set.
        p1, p2 = p
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            if p2>self.p2star:
                share = max(beta*(1-(1-self.p1star)/self.p1star*p1/(1-p1)), 0)
            else:
                share = 0
            x = Allocation(p, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            y = Allocation(p, {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':1}, self.mechanism)
            feasible = [x, y]
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            x = Allocation(p, {'l1':{'l2':1, 'h2':0}, 'h1':{'l2': 1, 'h2':0}, 'h_share':0}, self.mechanism)
            y = Allocation(p, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            z = Allocation(p, {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':1}, self.mechanism)
            feasible = [x, y, z]
        return feasible    

    def find_frontier(self, p):
        #Finds allocations on the strong Pareto frontier of the feasible sets (across self.player beliefs).
        p1, p2 = p
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            if p2>self.p2star:
                share = max(beta*(1-(1-self.p1star)/self.p1star*p1/(1-p1)), 0)
            else:
                share = 0
            xstar = Allocation([self.p1star, p2], {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            ystar = Allocation([self.p1star, p2], {'l1':{'l2':share, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':1}, self.mechanism)
            iAB = xstar.intersection(ystar, 'L', self.Ml(p))
            iC = xstar.intersection(ystar, 'H', self.Mh(p))
            frontier = [iAB, iC]
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            xstar = Allocation([p1, self.p2star], {'l1':{'l2':1, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            ystar = Allocation([p1, self.p2star], {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':0}, self.mechanism)
            zstar = Allocation([p1, self.p2star], {'l1':{'l2':0, 'h2':0}, 'h1':{'l2':1, 'h2':0}, 'h_share':1}, self.mechanism)
            iA = xstar.intersection(ystar, 'H', self.Mh(p))
            iB = ystar.intersection(zstar, 'L', self.Ml(p))
            iC = ystar.intersection(zstar, 'H', self.Mh(p))
            if p1<=self.p1star: #Zone A adn B:
                frontier = [iA, ystar, iB]
            if p1>=self.p1star: #Zone C and B:
                frontier = [iC, iB]
        return frontier

    def find_allocation(self, p):
        p1, p2 = p
        #Find the allocation in the mechanism
        zones = self.find_zones(p)
        feasible = self.find_feasible(p)
        frontier = self.find_frontier(p)
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            if p1 == self.p1star:
                allocation = frontier
            else: 
                x = feasible[0]
                y = feasible[1]
                iAB = x.intersection(y, 'L', self.Ml(p))
                iC = x.intersection(y, 'H', self.Mh(p))
                allocation = []
                if ('A' in zones or 'B' in zones): 
                    allocation.append(iAB)
                if 'C' in zones: #Zone C
                    allocation.append(iC)
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            if p2 == self.p2star:
                allocation = frontier
            else:
                x = feasible[0]
                y = feasible[1]
                z = feasible[2]
                iA = x.intersection(y, 'H', self.Mh(p))
                iB = y.intersection(z, 'L', self.Ml(p))
                iC = y.intersection(z, 'H', self.Mh(p))
                if 'A' in zones:
                    allocation = [iA]
                if 'B' in zones:
                    allocation = [iB]
                if 'C' in zones:
                    allocation = [iC]
        for a in allocation:
            a.set_player(self.player)
        return allocation

    def shade1(self, x):
        # Example function, replace with your actual function
        return  np.minimum((1-x)/(1-self.p1star), x/self.p1star) 
        
    def shade2(self, y):
        # Example function, replace with your actual function
        return np.minimum((1-y)/(1-self.p2star), (y-self.p2starstar)/(self.p2star - self.p2starstar))
        
    def shade(self, x, y):
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            return  self.shade1(x) * self.shade2(y)
        else: 
            return x-x
    
    def __repr__(self):
        return f"Mechanism({self.mechanism}, {self.case})"
    
    def name(self):
        return self.mechanism
    
    def find_linearly_transferable_zones(self):
        if self.mechanism == 'a1' and self.case == 'l2<h1':
            return ['A']
        if self.mechanism == 'a2' and self.case == 'l2<h1':
            return ['C']
    
    def payoffs(self, l, h, p):
        p1, p2 = p
        if self.player == '1':
            q = p1
        if self.player == '2':
            q = p2 
        if self.mechanism_payoff == 'expected':
            return (1-q)*l + q*h
        if self.mechanism_payoff == 'u(l)': 
            return l
        if self.mechanism_payoff == 'u(h)': 
            return h       
    
    def find_payoff_crossections(self, p):
        epsilon = 0.001
        p1, p2 = p
        allocations, monopolies = [], []
        current_allocation = self.find_allocation(p)[0]
        if self.mechanism == 'a2':
            current = [p2, self.payoffs(current_allocation.l(), current_allocation.h(), p)]
            ps = [epsilon, self.p2star-epsilon, self.p2star+epsilon, 1-epsilon]
            for q2 in ps:
                q = [p1, q2]
                allocation = self.find_allocation(q)[0]
                allocations.append(self.payoffs(allocation.l(), allocation.h(), p))
                monopolies.append(self.payoffs(self.Ml(q), self.Mh(q), p))
        if self.mechanism == 'a1':
            current = [p1, self.payoffs(current_allocation.l(), current_allocation.h(), p)]
            ps = [epsilon, self.p1star-epsilon, self.p1star+epsilon, 1-epsilon]
            for q1 in ps:
                q = [q1, p2]
                allocation = self.find_allocation(q)[0]
                allocations.append(self.payoffs(allocation.l(), allocation.h(), p))
                monopolies.append(self.payoffs(self.Ml(q), self.Mh(q), p))
        return ps, allocations, monopolies, current
    
    def find_payoff_equations(self, p):
        return payoff_equations[self.case][self.mechanism][self.find_zones(p)[0]]
        
