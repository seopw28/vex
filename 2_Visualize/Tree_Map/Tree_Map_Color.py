#%% Import libraries
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

#%% Define color master list
color_master = [
    {'name': 'Coral Red', 'rgb': (255/255, 127/255, 80/255), 'value': 25},
    {'name': 'Sage Green', 'rgb': (188/255, 208/255, 178/255), 'value': 20},
    {'name': 'Dusty Rose', 'rgb': (220/255, 182/255, 193/255), 'value': 18},
    {'name': 'Slate Blue', 'rgb': (112/255, 128/255, 144/255), 'value': 15},
    {'name': 'Terracotta', 'rgb': (204/255, 78/255, 92/255), 'value': 12},
    {'name': 'Mint Green', 'rgb': (152/255, 251/255, 152/255), 'value': 10},
    {'name': 'Lavender', 'rgb': (230/255, 230/255, 250/255), 'value': 8},
    {'name': 'Mustard Yellow', 'rgb': (255/255, 219/255, 88/255), 'value': 7},
    {'name': 'Steel Blue', 'rgb': (70/255, 130/255, 180/255), 'value': 5},
    {'name': 'Blush Pink', 'rgb': (255/255, 192/255, 203/255), 'value': 3}
]

#%% Sort and prepare data
# Sort colors by value (descending)
color_master.sort(key=lambda x: x['value'], reverse=True)

# Extract values and names
values = [color['value'] for color in color_master]
names = [color['name'] for color in color_master]
colors = [color['rgb'] for color in color_master]

# Calculate total value
total_value = sum(values)

#%% Define treemap creation function
def create_treemap(values, names, colors, ax, x=0, y=0, width=1, height=1):
    # Sort values in descending order
    sorted_indices = np.argsort(values)[::-1]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    # Calculate total value
    total = sum(sorted_values)
    
    # Initialize position
    pos_x = x
    pos_y = y
    remaining_width = width
    remaining_height = height
    
    # Create rectangles
    for i, (value, name, color) in enumerate(zip(sorted_values, sorted_names, sorted_colors)):
        # Calculate rectangle size
        if remaining_width > remaining_height:
            rect_width = (value / total) * width
            rect_height = remaining_height
        else:
            rect_width = remaining_width
            rect_height = (value / total) * height
        
        # Create rectangle
        rect = Rectangle((pos_x, pos_y), rect_width, rect_height, 
                         facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        
        # Add text
        ax.text(pos_x + rect_width/2, pos_y + rect_height/2, 
                f"{name}\n{value}", 
                ha='center', va='center', 
                color='black' if sum(color) > 1.5 else 'white',
                fontsize=10)
        
        # Update position
        if remaining_width > remaining_height:
            pos_x += rect_width
            remaining_width -= rect_width
        else:
            pos_y += rect_height
            remaining_height -= rect_height

#%% Create and display visualization
# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Top 10 Designer Colors Treemap', fontsize=16, pad=20)

# Create treemap
create_treemap(values, names, colors, ax, 0, 0, 1, 1)

# Set axis limits and remove ticks
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])

# Add a legend with RGB values
legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color['rgb'], edgecolor='white') 
                  for color in color_master]
legend_labels = [f"{color['name']}: RGB{tuple(int(x*255) for x in color['rgb'])}" 
                for color in color_master]
ax.legend(legend_elements, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout to make room for legend
plt.tight_layout()

# Save the figure
plt.savefig('Colors_Treemap.png', dpi=300, bbox_inches='tight')
plt.show() 
# %%
