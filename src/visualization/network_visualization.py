import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def visualize_network(graph, figsize=(12, 10), node_size=50, 
                     opinion_colormap='RdBu_r', layout='auto'):
    """
    Visualize network with bots as squares and humans as circles.
    
    Parameters:
    - graph: igraph Graph object
    - figsize: tuple for figure size
    - node_size: base size for nodes
    - opinion_colormap: colormap for opinion coloring
    - layout: 'auto', 'fr', 'kk', or custom layout
    """
    
    # Get layout
    if layout == 'auto':
        pos = graph.layout_auto()
    elif layout == 'fr':
        pos = graph.layout_fruchterman_reingold()
    elif layout == 'kk':
        pos = graph.layout_kamada_kawai()
    else:
        pos = layout
    
    # Extract coordinates
    x_coords = [p[0] for p in pos]
    y_coords = [p[1] for p in pos]
    
    # Get node attributes
    opinions = graph.vs['opinion']
    is_bot = graph.vs['is_bot']
    
    # Setup color mapping for opinions
    norm = Normalize(vmin=-1, vmax=1)
    cmap = cm.get_cmap(opinion_colormap)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges first (so they appear behind nodes)
    for edge in graph.es:
        start_idx = edge.source
        end_idx = edge.target
        ax.plot([x_coords[start_idx], x_coords[end_idx]], 
                [y_coords[start_idx], y_coords[end_idx]], 
                'k-', alpha=0.3, linewidth=0.5)
    
    # Draw nodes
    for i in range(graph.vcount()):
        x, y = x_coords[i], y_coords[i]
        opinion = opinions[i]
        color = cmap(norm(opinion))
        
        if is_bot[i]:
            # Draw bot as square
            square = patches.Rectangle((x - node_size/200, y - node_size/200), 
                                     node_size/100, node_size/100,
                                     facecolor=color, edgecolor='black', 
                                     linewidth=1.5)
            ax.add_patch(square)
        else:
            # Draw human as circle
            circle = patches.Circle((x, y), radius=node_size/200,
                                  facecolor=color, edgecolor='black',
                                  linewidth=0.8)
            ax.add_patch(circle)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Opinion', rotation=270, labelpad=20)
    
    # Customize plot - add more margin to prevent node clipping
    margin = max(node_size/100, 0.15)
    ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend with proper shapes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, markeredgecolor='black', label='Human'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=8, markeredgecolor='black', label='Bot')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax