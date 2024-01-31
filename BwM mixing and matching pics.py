# To activate virtual environment, run .venv\Scripts\activate.ps1
# After activating virtual environment, make sure that requiremnts are installed by running: pip install -r requirements.txt
# To test run it locally, run: python -m streamlit run BwMpayoffs.py
# To stop the local server, press Ctrl+C (in the terminal window)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
        
p1star = 0.6

def plot_mixing(ax, q1=p1star):
    
    #ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.tick_params(bottom=False, left=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([-3.5, 9])  # For example, x-axis from 0 to 5
    ax.set_ylim([-3.5, 9]) # For example, y-axis from 0 to 20
    ax.grid(False)
    if q1 < p1star:
        title = f'$q_1<p_1^*$'
    elif q1 == p1star:
        title = f'$q_1=p_1^*$'
    else:
        title = f'$q_1>p_1^*$'

    #ax.plot([0, 10], [0, 0], color='black', linestyle='-')
    #ax.plot([0, 0], [0, 10], color='black', linestyle='-')
    ax.text(3, 0, '$u(l_1)$', horizontalalignment='center', verticalalignment='top')
    ax.text(0, 4, '$u(h_1)$', horizontalalignment='right', verticalalignment='center')
    ax.set_title(title)

    #Draw a_2 and a_2-w
    ax.plot(0, 0, marker = 'o', color='green')
    ax.text(0, 0, f'$a_1$', color='green', horizontalalignment='left', verticalalignment='top')
    ax.fill_between([0, 10], [0, 0], [10, 10], color="none", facecolor="green", alpha = 0.1, linewidth=0.0)
    if q1 < p1star:
        ax.fill_between([-3, 10], [5, 5], [10, 10], color="none", facecolor="blue", alpha = 0.1, linewidth=0.0)
        ax.plot(0, 8, marker = 'o', color='grey')
        ax.plot(-3, 5, marker = 'o', color='blue')
        #ax.plot(0, 5, marker = 'o', color='orange')
        ax.text(0, 8, f'$a_2(q_1,p_2)$', color='grey', horizontalalignment='left', verticalalignment='center')
        ax.text(-3, 5, f'$a_2(q_1,p_2)-w(q_1)$', color='blue', horizontalalignment='center', verticalalignment='top')
    if q1 == p1star:
        ax.fill_between([0, 0.75, 10], [1, 0, 0], [10, 10, 10], color="none", facecolor="blue", alpha = 0.1, linewidth=0.0)
        ax.plot(-3, 5, marker = 'o', color='blue')
        ax.plot(3, -3, marker = 'o', color='blue')
        ax.plot(0, 8, marker = 'o', color='grey')
        ax.plot(6, 0, marker = 'o', color='grey')
        ax.plot([0, 6], [8, 0], color='grey', linestyle='-')
        ax.plot([-3, 3], [5, -3], color='blue', linestyle='-')
    if q1 > p1star:
        ax.fill_between([3, 10], [-3, -3], [10, 10], color="none", facecolor="blue", alpha = 0.1, linewidth=0.0)
        ax.plot(6, 0, marker = 'o', color='grey')
        #ax.plot(3, 0, marker = 'o', color='orange')
        ax.plot(3, -3, marker = 'o', color='blue')
        ax.text(6,0, f'$a_2(q_1,p_2)$', color='grey', horizontalalignment='center', verticalalignment='top')
        ax.text(3, -3, f'$a_2(q_1,p_2)-w(q_1)$', color='blue', horizontalalignment='center', verticalalignment='top')

def draw():

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_mixing(axs[0], 0.3)
    plot_mixing(axs[1], 0.6)
    plot_mixing(axs[2], 0.75)

    plt.tight_layout()
    plt.savefig('figures/BwMmixing.png', dpi=600)
    plt.show()
    

draw()