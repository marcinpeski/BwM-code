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

payoff_equations = {'l2<h1':{\
                'a1':{'A':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)+\left(1-x\right)p_{1}\frac{h_{2}-h_{1}}{h_{2}-l_{1}}\left(l_{2}-l_{1}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$',}, \
                    'B':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)+x\left(1-p_{1}\right)\left(h_{2}-l_{2}\right)$',}, \
                    'C':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)+x\left(1-p_{1}\right)\left(l_{2}-l_{1}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'}},

                'a2':{'A':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)+\left(1-x\right)\left(1-p_{2}\right)\frac{h_{2}-h_{1}}{h_{2}-l_{2}}\left(l_{2}-l_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'}, \
                    'B':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)+\left(1-x\right)p_{2}\left(h_{2}-h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right) - \left(1-x\right)p_{1}\left(h_{2}-h_{1}\right) + x\left(1-p_{1}\right)\left(h_{2}-l_{2}\right)$'}, \
                    'C':{'l1':r'$x\left(\left(1-p_{2}\right) l_{2}+ p_{2}h_{2} \right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'}}},

           'l2>h1':{\
                'a1':{'A':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)+\left(1-x\right)p_{1}\frac{h_{2}-l_{2}}{h_{2}-l_{1}}\left(h_{1}-l_{1}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$',}, \
                    'B':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)+x\left(1-p_{1}\right)\left(h_{2}-l_{2}\right)$'}, \
                    'C':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)+x\left(1-p_{1}\right)\left(h_{1}-l_{1}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'},\
                    'D':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)+x\left(h_{2}-l_{2}\right)$'}},

                'a2':{'A':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)+\left(1-x\right)\left(1-p_{2}\right)\left(h_{1}-l_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'}, \
                    'B':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)+\left(1-x\right)\left(p_{1}h_{2}+\letf(1-p_{2}\right)h_{1}-l_{2}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right) + \left(x-p_{1}\right)\left(h_{2}-l_{2}\right)$'}, \
                    'C':{'l1':r'$x\left( \left(1-p_{2}\right)h_{1} + p_{2}h_{2} \right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right)$'},\
                    'D':{'l1':r'$xM_{1}\left(l_{1}\right)$', \
                        'h1':r'$xM_{1}\left(h_{1}\right)$', \
                        'l2':r'$\left(1-x\right)M_{2}\left(l_{2}\right)$', \
                        'h2':r'$\left(1-x\right)M_{2}\left(h_{2}\right) + x\left(h_{2}-l_{2}\right)$'}}}}
