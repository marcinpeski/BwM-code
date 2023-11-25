import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title('Interactive Trigonometric Plots')

# Sliders
alpha = st.slider('Select Alpha', 0.1, 5.0, 1.0, 0.1)
d = st.slider('Select d', 0.0, 2 * np.pi, 0.0, 0.1)

# Plotting Function
def plot_trig_functions(alpha, d):
    x = np.linspace(-10, 10, 400)
    y_sin = np.sin(alpha * x + d)
    y_cos = np.cos(alpha * x + d)

    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    ax[0].plot(x, y_sin)
    ax[0].set_title('sin(αx + d)')
    ax[0].grid(True)

    ax[1].plot(x, y_cos)
    ax[1].set_title('cos(αx + d)')
    ax[1].grid(True)

    return fig

# Display the plot
fig = plot_trig_functions(alpha, d)
st.pyplot(fig)