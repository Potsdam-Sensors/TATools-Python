import numpy as np
import matplotlib.pyplot as plt
from TATools.plotting.figures import multi_yaxis_figure
# %matplotlib widget  # uncomment if you prefer ipympl

n_yaxes = {{n_yaxes|3}}
colors = {{colors|["blue", "orange", "green"]}}
figsize = {{figsize|(13,5)}}

fig, axes = multi_yaxis_figure(n_yaxes, colors, figsize)

# Placeholder Data -- Feel free to delete
x_data = np.linspace(0, 2*np.pi, 100)
coeffs_y = list(range(n_yaxes))
for color, axis, coef in zip(colors, axes, coeffs_y): # There are >=`n_yaxes` of each color, axes, and coeffs_y -- zip them together and iterate all three at once
    axis.plot(np.sin((1+coef)*x_data)*(2+coef), color=color)
    axis.set_ylabel(f"This line is {color}")