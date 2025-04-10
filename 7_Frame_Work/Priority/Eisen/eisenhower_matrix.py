#%%
import matplotlib.pyplot as plt
import numpy as np

# Sample tasks for each quadrant
tasks = {
    'Quadrant I (Urgent & Important)': [
        'Prepare client presentation',
        'Fix critical bug',
        'Handle customer complaint'
    ],
    'Quadrant II (Not Urgent & Important)': [
        'Plan next quarter strategy',
        'Team training',
        'Personal development'
    ],
    'Quadrant III (Urgent & Not Important)': [
        'Respond to non-critical emails',
        'Attend unnecessary meetings',
        'Handle minor issues'
    ],
    'Quadrant IV (Not Urgent & Not Important)': [
        'Browse social media',
        'Watch random videos',
        'Gossip with colleagues'
    ]
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Draw the matrix lines
ax.axvline(x=0.5, color='black', linestyle='-', linewidth=2)
ax.axhline(y=0.5, color='black', linestyle='-', linewidth=2)

# Set axis limits and remove ticks
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])

# Add quadrant labels with adjusted positions
ax.text(0.25, 0.95, 'Quadrant I\nUrgent & Important', 
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.75, 0.95, 'Quadrant II\nNot Urgent & Important', 
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.25, 0.40, 'Quadrant III\nUrgent & Not Important', 
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.75, 0.40, 'Quadrant IV\nNot Urgent & Not Important', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Add tasks to each quadrant
y_positions = [0.7, 0.65, 0.6]  # Adjusted vertical positions for tasks
for i, (quadrant, task_list) in enumerate(tasks.items()):
    x = 0.25 if i < 2 else 0.75
    y_start = 0.8 if i % 2 == 0 else 0.3
    
    for j, task in enumerate(task_list):
        y = y_start - j * 0.1
        ax.text(x, y, f'• {task}', ha='center', va='center', fontsize=10)

# Add axis labels
ax.set_xlabel('Urgency', fontsize=14, fontweight='bold')
ax.set_ylabel('Importance', fontsize=14, fontweight='bold')

# Add title
plt.title('Eisenhower Matrix: Task Prioritization', fontsize=16, fontweight='bold', pad=20)

# Add color to quadrants
ax.fill_between([0, 0.5], [0.5, 0.5], [1, 1], color='#FF9999', alpha=0.3)  # Q1
ax.fill_between([0.5, 1], [0.5, 0.5], [1, 1], color='#99FF99', alpha=0.3)  # Q2
ax.fill_between([0, 0.5], [0, 0], [0.5, 0.5], color='#9999FF', alpha=0.3)  # Q3
ax.fill_between([0.5, 1], [0, 0], [0.5, 0.5], color='#FFFF99', alpha=0.3)  # Q4

# Adjust layout and save
plt.tight_layout()
plt.savefig('eisenhower_matrix.png', dpi=300, bbox_inches='tight')
plt.show() 
# %%
