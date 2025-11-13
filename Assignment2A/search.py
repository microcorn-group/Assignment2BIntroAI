import sys
import math
from collections import deque
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.patches import FancyBboxPatch
from bst_visualizer import (BST, create_bst_from_all_nodes,
                            calculate_tree_layout, setup_bst_visualization, highlight_node)

# ------------------------------
# LOAD GRAPH DATA
# ------------------------------
def load_problem(filename):
    nodes, edges, origin, destinations = {}, {}, None, []
    section = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(('Nodes:', 'Edges:', 'Origin:', 'Destinations:')):
                section = line.split(':')[0].lower()
                continue

            if section == 'nodes':
                node, coords = line.split(':')
                x, y = coords.strip(" ()").split(',')
                nodes[node.strip()] = (float(x), float(y))
            elif section == 'edges':
                left, cost = line.split(':')
                a, b = left.strip(" ()").split(',')
                edges.setdefault(a.strip(), []).append((b.strip(), float(cost.strip())))
            elif section == 'origin':
                origin = line.strip()
            elif section == 'destinations':
                destinations = [d.strip() for d in line.split(";")]

    return nodes, edges, origin, destinations

# ------------------------------
# HEURISTIC FUNCTION
# ------------------------------
def heuristic(a, b, nodes):
    (x1, y1) = nodes[a]
    (x2, y2) = nodes[b]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# ------------------------------
# PRINT RESULTS
# ------------------------------
def print_results(filename, method, goal, num_nodes, path):
    print(f"{filename} {method}")
    print(f"{goal} {num_nodes}")
    print(" -> ".join(path))

# ------------------------------
# INTERACTIVE VISUALIZATION WITH UI CONTROLS (+ BST SIDE-BY-SIDE)
# ------------------------------
def visualize_search(nodes, edges, visited_order, path, method_name, start, goals, bst=None, filename=None):
    # Create figure with extra space for controls and improved layout
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor('#F8F9FA')  # Slightly improved gray
    
    # Add main title at the top with more space
    fig.suptitle('Interactive Pathfinding Algorithm Visualizer', 
                 fontsize=22, fontweight='bold', color='#1976D2', y=0.99)
    
    # Main graph axes on the LEFT with more vertical space
    if bst:
        # If BST is provided, use left side for graph and right side for BST
        ax = plt.subplot2grid((18, 24), (1, 0), colspan=12, rowspan=9)
    else:
        # If no BST, use full width
        ax = plt.subplot2grid((18, 16), (1, 0), colspan=16, rowspan=9)
    ax.set_facecolor('#FFFFFF')
    ax.set_xlabel("X Coordinate", fontsize=11, fontweight='bold', color='#555555')
    ax.set_ylabel("Y Coordinate", fontsize=11, fontweight='bold', color='#555555')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.7, color='#E0E0E0')
    ax.set_title("Search Graph", fontsize=13, fontweight='bold', color='#424242', pad=10)
    
    # Set axis limits with padding
    x_coords = [coord[0] for coord in nodes.values()]
    y_coords = [coord[1] for coord in nodes.values()]
    x_margin = (max(x_coords) - min(x_coords)) * 0.2 or 1
    y_margin = (max(y_coords) - min(y_coords)) * 0.2 or 1
    ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
    ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
    
    # Add subtle border to the plot area
    for spine in ax.spines.values():
        spine.set_edgecolor('#BBBBBB')
        spine.set_linewidth(1.5)

    # Initialize BST data early
    bst_data = {}

    # State management
    state = {
        'current_method': method_name,
        'animation': None,
        'is_playing': False,
        'current_frame': 0,
        'path_lines': [],
        'exploration_lines': [],  # Lines showing the exploration path (red/orange)
        'edge_lines': [],
        'scatters': {},
        'visit_labels': {},
        'node_labels': {},
        'bst_data': bst_data  # Will be updated later if BST exists
    }

    # --- Draw edges and edge costs ---
    for src, neighbors in edges.items():
        x1, y1 = nodes[src]
        for dest, cost in neighbors:
            x2, y2 = nodes[dest]
            line, = ax.plot([x1, x2], [y1, y2], color='#A8A8A8', linestyle='-', 
                           linewidth=2, zorder=1, alpha=0.5)
            state['edge_lines'].append(line)
            
            midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(midx, midy, f"{cost:.1f}", fontsize=10, color='#2E7D32', 
                   fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8F5E9', 
                            edgecolor='#4CAF50', alpha=0.9, linewidth=1.5), zorder=2)

    # Color scheme constants
    COLORS = {
        'start': ('#4CAF50', '#2E7D32', 900, 3.5),
        'goal': ('#F44336', '#C62828', 900, 3.5),
        'normal': ('#ECEFF1', '#78909C', 750, 2.5),
    }
    
    # --- Draw nodes ---
    for node, (x, y) in nodes.items():
        if node == start:
            color, edge_color, size, edge_width = COLORS['start']
        elif node in goals:
            color, edge_color, size, edge_width = COLORS['goal']
        else:
            color, edge_color, size, edge_width = COLORS['normal']
            
        scatter = ax.scatter(x, y, color=color, s=size, zorder=5, 
                            edgecolors=edge_color, linewidths=edge_width, alpha=0.95)
        state['scatters'][node] = scatter
        
        # Node label
        label = ax.text(x, y, f"{node}", fontsize=15, ha='center', va='center', 
                       fontweight='bold', color='#212121', zorder=6)
        state['node_labels'][node] = label
        
        # Visit order label (initially empty)
        visit_label = ax.text(x, y - 0.5, "", fontsize=9, ha='center', 
                             va='top', color='#1976D2', fontweight='bold', zorder=6,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', 
                                      edgecolor='#2196F3', alpha=0, linewidth=1))
        state['visit_labels'][node] = visit_label

    # --- Draw BST on RIGHT SIDE if provided ---
    bst_data = {}  # Store BST visualization data
    if bst:
        from bst_visualizer import calculate_tree_layout, draw_tree_edges
        import matplotlib.patches as patches
        
        ax_bst = plt.subplot2grid((18, 24), (1, 12), colspan=12, rowspan=9)
        ax_bst.set_facecolor('#FFFFFF')  # Clean white background
        ax_bst.set_aspect('equal')
        ax_bst.axis('off')
        
        # Title
        ax_bst.text(0.5, 0.98, 'Search Tree', 
                   transform=ax_bst.transAxes, fontsize=13, fontweight='bold',
                   ha='center', va='top', color='#424242')
        
        # Calculate BST layout
        bst_positions = calculate_tree_layout(bst.root)
        bst_positions_dict = {value: (x, y) for value, x, y in bst_positions}
        
        # Set up tree visualization (draw all nodes as inactive/gray first)
        node_circles, node_texts, node_badges = setup_bst_visualization(ax_bst, bst, bst_positions_dict)
        bst_data = {
            'ax': ax_bst,
            'positions_dict': bst_positions_dict,
            'node_circles': node_circles,
            'node_texts': node_texts,
            'node_badges': node_badges
        }
        
        # Update state with BST data
        state['bst_data'] = bst_data
        
        # Set axis limits
        if bst_positions:
            xs = [pos[1] for pos in bst_positions]
            ys = [pos[2] for pos in bst_positions]
            margin = 300
            ax_bst.set_xlim(min(xs) - margin, max(xs) + margin)
            ax_bst.set_ylim(min(ys) - margin, max(ys) + margin)

    # Info box with unified styling
    info_box = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                      verticalalignment='top', family='monospace',
                      bbox=dict(boxstyle='round,pad=0.7', facecolor='#E3F2FD', 
                               edgecolor='#1976D2', alpha=0.92, linewidth=1.5),
                      fontweight='bold', color='#1565C0')

    # --- Legend with enhanced styling ---
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', 
                  markersize=12, markeredgecolor='#2E7D32', markeredgewidth=2.5,
                  label='Start Node', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', 
                  markersize=12, markeredgecolor='#C62828', markeredgewidth=2.5,
                  label='Goal Node', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', 
                  markersize=12, markeredgecolor='#F57C00', markeredgewidth=2.5,
                  label='Explored', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ECEFF1', 
                  markersize=12, markeredgecolor='#78909C', markeredgewidth=2.5,
                  label='Unvisited', linestyle='None'),
        plt.Line2D([0], [0], color='#E53935', linewidth=3.5, label='Exploration Path'),
        plt.Line2D([0], [0], color='#2E7D32', linewidth=4, label='Final Path'),
    ]
    legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=10, 
                      framealpha=0.95, edgecolor='#BDBDBD', fancybox=True,
                      shadow=False, title='Legend', title_fontsize=11,
                      bbox_to_anchor=(0.0, -0.52))
    legend.get_frame().set_facecolor('#FAFAFA')
    legend.get_frame().set_linewidth(1.5)

    # --- Button axes (at bottom with better spacing) ---
    if bst:
        button_row = 16
        progress_row = 15
        grid_cols = (18, 24)
        btn_cols = 24
    else:
        button_row = 16
        progress_row = 15
        grid_cols = (18, 16)
        btn_cols = 16
    
    # --- Progress bar (video-like) ---
    ax_progress = plt.subplot2grid(grid_cols, (progress_row, 0), colspan=btn_cols, rowspan=1)
    ax_progress.set_facecolor('#E0E0E0')
    ax_progress.set_xlim(0, 1)
    ax_progress.set_ylim(0, 1)
    ax_progress.axis('off')
    
    # Add border to progress bar
    progress_bar_rect = plt.Rectangle((0.02, 0.2), 0.96, 0.6, 
                                      fill=False, edgecolor='#424242', 
                                      linewidth=2, transform=ax_progress.transAxes)
    ax_progress.add_patch(progress_bar_rect)
    
    # Progress fill (will be updated)
    progress_fill = plt.Rectangle((0.02, 0.2), 0, 0.6, 
                                  fill=True, facecolor='#1976D2', 
                                  alpha=0.8, transform=ax_progress.transAxes)
    ax_progress.add_patch(progress_fill)
    
    # Progress text in the middle
    progress_text = ax_progress.text(0.5, 0.5, '0%', 
                                    transform=ax_progress.transAxes,
                                    ha='center', va='center',
                                    fontsize=12, fontweight='bold',
                                    color='#000000', zorder=10)
    
    state['progress_fill'] = progress_fill
    state['progress_text'] = progress_text

    # --- Button controls ---
    btn_data = [('Back', 0, '#1976D2', '#0D47A1'), ('Pause', 4, '#F57C00', '#E65100'),
                ('Forward', 8, '#1976D2', '#0D47A1'), ('Restart', 12, '#388E3C', '#1B5E20')]
    
    buttons = {}
    for label, col, color, hover in btn_data:
        ax_btn = plt.subplot2grid(grid_cols, (button_row, col), colspan=3, rowspan=1)
        btn = Button(ax_btn, label, color=color, hovercolor=hover)
        btn.label.set_fontsize(11)
        btn.label.set_fontweight('bold')
        btn.label.set_color('#FFFFFF')
        # Add some styling to the button
        ax_btn.spines['top'].set_visible(False)
        ax_btn.spines['right'].set_visible(False)
        ax_btn.spines['bottom'].set_linewidth(2)
        ax_btn.spines['left'].set_linewidth(2)
        buttons[label] = btn

    # Reset visualization
    def reset_viz():
        # Clear path lines
        for line in state['path_lines']:
            line.remove()
        state['path_lines'].clear()
        
        # Clear exploration lines
        for line in state['exploration_lines']:
            line.remove()
        state['exploration_lines'].clear()
        
        # Reset node colors using COLORS constant
        for node in state['scatters']:
            if node == start:
                color, edge_color, _, _ = COLORS['start']
            elif node in goals:
                color, edge_color, _, _ = COLORS['goal']
            else:
                color, edge_color, _, _ = COLORS['normal']
            state['scatters'][node].set_color(color)
            state['scatters'][node].set_edgecolor(edge_color)
        
        # Clear visit labels
        for label in state['visit_labels'].values():
            label.set_text("")
            label.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', 
                               edgecolor='#2196F3', alpha=0, linewidth=1))
        
        # Reset BST tree nodes to inactive state
        if state['bst_data']:
            for node_val in state['bst_data'].get('node_circles', {}):
                state['bst_data']['node_circles'][node_val].set_facecolor('#F0F0F0')
                state['bst_data']['node_circles'][node_val].set_edgecolor('#CCCCCC')
                state['bst_data']['node_circles'][node_val].set_linewidth(2.5)
                state['bst_data']['node_texts'][node_val].set_color('#999999')
                if node_val in state['bst_data'].get('node_badges', {}):
                    badge = state['bst_data']['node_badges'][node_val]
                    badge.set_text("")
                    if bbox := badge.get_bbox_patch():
                        bbox.set_alpha(0)
        
        state['current_frame'] = 0
        info_box.set_text("")
        ax.set_title("", fontsize=15, fontweight='bold', pad=20, color='#424242')
        fig.canvas.draw_idle()

    # Function to manually draw a specific frame (for stepping)
    def draw_frame(frame, visited_order, path, method):
        """Manually draw a specific frame without triggering final path logic"""
        state['current_frame'] = frame
        
        # Extract just the node names from visited_order (which may contain tuples)
        visited_nodes = [v[0] if isinstance(v, tuple) else v for v in visited_order]
        
        # First, clear and show all visited nodes up to this frame
        for idx, node in enumerate(visited_nodes):
            if idx < frame:
                # This node should be shown as visited
                if node not in (start, *goals):
                    state['scatters'][node].set_color('#FF9800')
                    state['scatters'][node].set_edgecolor('#F57C00')
                    state['scatters'][node].set_alpha(0.95)
                
                state['visit_labels'][node].set_text(f"#{idx + 1}")
                state['visit_labels'][node].set_bbox(dict(boxstyle='round,pad=0.3', 
                                                         facecolor='#E3F2FD', 
                                                         edgecolor='#2196F3', 
                                                         alpha=0.9, linewidth=1))
                
                # Highlight in BST
                if state['bst_data']:
                    highlight_node(state['bst_data']['node_circles'],
                                  state['bst_data']['node_texts'],
                                  state['bst_data']['node_badges'],
                                  node, idx + 1)
            else:
                # This node should not be highlighted yet
                if node not in (start, *goals):
                    state['scatters'][node].set_color('#ECEFF1')
                    state['scatters'][node].set_edgecolor('#78909C')
                    state['scatters'][node].set_alpha(0.95)
                
                state['visit_labels'][node].set_text("")
                state['visit_labels'][node].set_bbox(dict(boxstyle='round,pad=0.3', 
                                                         facecolor='#E3F2FD', 
                                                         edgecolor='#2196F3', 
                                                         alpha=0, linewidth=1))
        
        # Draw exploration edges (red lines connecting visited nodes - using parent information)
        for line in state['exploration_lines']:
            line.remove()
        state['exploration_lines'].clear()
        
        # Draw all exploration edges up to the current frame
        for idx in range(min(frame, len(visited_order))):
            node_info = visited_order[idx]
            # visited_order contains (node, parent) tuples
            if isinstance(node_info, tuple):
                node, parent = node_info
            else:
                node = node_info
                parent = None
            
            # Draw line from parent to this node
            if parent is not None and parent in nodes and node in nodes:
                x1, y1 = nodes[parent]
                x2, y2 = nodes[node]
                line, = ax.plot([x1, x2], [y1, y2], color='#E53935', 
                              linewidth=2.5, zorder=3, alpha=0.6, 
                              solid_capstyle='round')
                state['exploration_lines'].append(line)
        
        # Update info box for searching state
        if frame < len(visited_nodes):
            current = visited_nodes[frame]
            progress = (frame / len(visited_nodes)) * 100
            progress_bar = '█' * int(progress / 10) + '░' * (10 - int(progress / 10))
            
            # Update video-like progress bar
            progress_ratio = frame / len(visited_nodes)
            state['progress_fill'].set_width(0.96 * progress_ratio)
            state['progress_text'].set_text(f'{progress:.0f}%')
            
            info_box.set_text(f"Algorithm: {method}\n"
                            f"Nodes Explored: {frame} / {len(visited_nodes)}\n"
                            f"Current Node: {current}\n"
                            f"Status: Searching...")
            
            ax.set_title(f"Path Finding Visualization — {method} Algorithm", 
                        fontsize=15, fontweight='bold', pad=20, color='#1976D2')
        
        elif frame >= len(visited_nodes):
            # Update progress bar to 100%
            state['progress_fill'].set_width(0.96)
            state['progress_text'].set_text('100%')
            state['progress_fill'].set_facecolor('#4CAF50')
            
            # Show final path in green (on top of exploration path)
            for line in state['path_lines']:
                line.remove()
            state['path_lines'].clear()
            
            # Calculate total path cost and draw final path in green
            path_cost = 0
            for i in range(len(path) - 1):
                x1, y1 = nodes[path[i]]
                # Draw green line for the final optimal path (thicker, on top)
                line, = ax.plot([x1, x2], [y1, y2], color='#2E7D32', 
                              linewidth=6, zorder=5, alpha=0.95, 
                              solid_capstyle='round', label='Final Path' if i == 0 else '')
                state['path_lines'].append(line)
                
                # Calculate cost for this edge
                for neighbor, cost in edges.get(path[i], []):
                    if neighbor == path[i + 1]:
                        path_cost += cost
                        break
            
            progress_bar = '█' * 10
            visited_nodes_count = len([v[0] if isinstance(v, tuple) else v for v in visited_order])
            info_box.set_text(f"Algorithm: {method}\n"
                            f"Progress: [{progress_bar}] 100%\n"
                            f"Nodes Explored: {visited_nodes_count}\n"
                            f"Path Length: {len(path)} nodes\n"
                            f"Total Cost: {path_cost:.1f}\n"
                            f"Status: COMPLETE!")
            info_box.set_bbox(dict(boxstyle='round,pad=0.8', facecolor='#C8E6C9', 
                                  edgecolor='#4CAF50', alpha=0.95, linewidth=2.5))
            
            ax.set_title(f"Path Finding Visualization — {method} Algorithm (COMPLETE)", 
                        fontsize=15, fontweight='bold', pad=20, color='#2E7D32')

    # Animation update function with enhanced visuals
    def update(frame, visited_order, path, method):
        state['current_frame'] = frame
        
        # Extract just the node names from visited_order (which may contain tuples)
        visited_nodes = [v[0] if isinstance(v, tuple) else v for v in visited_order]
        
        if frame < len(visited_nodes):
            current = visited_nodes[frame]
            visit_num = frame + 1
            
            if current not in (start, *goals):
                state['scatters'][current].set_color('#FF9800')
                state['scatters'][current].set_edgecolor('#F57C00')
                state['scatters'][current].set_alpha(0.95)
            
            state['visit_labels'][current].set_text(f"#{visit_num}")
            # Make the visit label background visible when visited
            state['visit_labels'][current].set_bbox(dict(boxstyle='round,pad=0.3', 
                                                         facecolor='#E3F2FD', 
                                                         edgecolor='#2196F3', 
                                                         alpha=0.9, linewidth=1))
            
            # ANIMATE BST: Light up the node in the search tree
            if state['bst_data']:
                highlight_node(state['bst_data']['node_circles'],
                              state['bst_data']['node_texts'],
                              state['bst_data']['node_badges'],
                              current, visit_num)
            
            # Draw exploration edges (red lines connecting visited nodes as we explore)
            # Use parent information from visited_order tuple (node, parent)
            if frame > 0:
                current_info = visited_order[frame - 1]
                if isinstance(current_info, tuple):
                    node, parent = current_info
                else:
                    node = current_info
                    parent = None
                
                # Draw line from parent to this node
                if parent is not None and parent in nodes and node in nodes:
                    x1, y1 = nodes[parent]
                    x2, y2 = nodes[node]
                    line, = ax.plot([x1, x2], [y1, y2], color='#E53935', 
                                  linewidth=2.5, zorder=3, alpha=0.6, 
                                  solid_capstyle='round')
                    state['exploration_lines'].append(line)
            
            # Calculate progress percentage
            progress = (visit_num / len(visited_nodes)) * 100
            progress_bar = '█' * int(progress / 10) + '░' * (10 - int(progress / 10))
            
            # Update video-like progress bar
            progress_ratio = visit_num / len(visited_nodes)
            state['progress_fill'].set_width(0.96 * progress_ratio)
            state['progress_text'].set_text(f'{progress:.0f}%')
            
            info_box.set_text(f"Algorithm: {method}\n"
                            f"Nodes Explored: {visit_num} / {len(visited_nodes)}\n"
                            f"Current Node: {current}\n"
                            f"Status: Searching...")
            
            ax.set_title(f"Path Finding Visualization — {method} Algorithm", 
                        fontsize=15, fontweight='bold', pad=20, color='#1976D2')
            
        elif frame == len(visited_nodes):
            # Draw final path in green on top of exploration path (don't clear exploration lines!)
            for line in state['path_lines']:
                line.remove()
            state['path_lines'].clear()
            
            # Calculate total path cost and draw final path in green
            path_cost = 0
            for i in range(len(path) - 1):
                x1, y1 = nodes[path[i]]
                x2, y2 = nodes[path[i + 1]]
                # Draw green line for the final optimal path (thicker, on top)
                line, = ax.plot([x1, x2], [y1, y2], color='#2E7D32', 
                              linewidth=6, zorder=5, alpha=0.95, 
                              solid_capstyle='round')
                state['path_lines'].append(line)
                
                # Calculate cost for this edge
                for neighbor, cost in edges.get(path[i], []):
                    if neighbor == path[i + 1]:
                        path_cost += cost
                        break
            
            visited_nodes_count = len([v[0] if isinstance(v, tuple) else v for v in visited_order])
            info_box.set_text(f"Algorithm: {method}\n"
                            f"Nodes Explored: {visited_nodes_count}\n"
                            f"Path Length: {len(path)} nodes\n"
                            f"Total Cost: {path_cost:.1f}\n"
                            f"Status: COMPLETE!")
            info_box.set_bbox(dict(boxstyle='round,pad=0.8', facecolor='#C8E6C9', 
                                  edgecolor='#4CAF50', alpha=0.95, linewidth=2.5))
            
            ax.set_title(f"Path Finding Visualization — {method} Algorithm (COMPLETE)", 
                        fontsize=15, fontweight='bold', pad=20, color='#2E7D32')
        
        return [info_box]

    # Button callbacks
    def on_back(event):
        """Step backward through the animation"""
        if state['animation'] and state['animation'].event_source:
            state['animation'].event_source.stop()
        state['is_playing'] = False
        buttons['Pause'].label.set_text('Resume')
        
        # Calculate the previous frame
        visited_nodes = [v[0] if isinstance(v, tuple) else v for v in visited_order]
        max_frame = len(visited_nodes) + 15
        new_frame = max(0, state['current_frame'] - 1)
        
        # Use the dedicated frame drawing function
        reset_viz()
        draw_frame(new_frame, visited_order, path, method_name)
        fig.canvas.draw_idle()
    
    def on_forward(event):
        """Step forward through the animation"""
        if state['animation'] and state['animation'].event_source:
            state['animation'].event_source.stop()
        state['is_playing'] = False
        buttons['Pause'].label.set_text('Resume')
        
        # Calculate the next frame
        visited_nodes = [v[0] if isinstance(v, tuple) else v for v in visited_order]
        max_frame = len(visited_nodes) + 15
        new_frame = min(max_frame - 1, state['current_frame'] + 1)
        
        # Use the dedicated frame drawing function
        reset_viz()
        draw_frame(new_frame, visited_order, path, method_name)
        fig.canvas.draw_idle()

    def on_restart(event):
        """Restart the animation from the beginning"""
        if state['animation'] and state['animation'].event_source:
            state['animation'].event_source.stop()
        state['is_playing'] = False
        reset_viz()
        
        # Restart the animation
        state['is_playing'] = True
        buttons['Pause'].label.set_text('Pause')
        state['animation'] = animation.FuncAnimation(
            fig, update, fargs=(visited_order, path, method_name),
            frames=len(visited_order) + 15, interval=600, repeat=False, blit=False)
        fig.canvas.draw_idle()

    def on_pause(event):
        if state['animation'] and state['animation'].event_source:
            if state['is_playing']:
                state['animation'].event_source.stop()
                state['is_playing'] = False
                buttons['Pause'].label.set_text('Resume')
            else:
                state['animation'].event_source.start()
                state['is_playing'] = True
                buttons['Pause'].label.set_text('Pause')
            fig.canvas.draw_idle()

    # Connect buttons
    buttons['Back'].on_clicked(on_back)
    buttons['Forward'].on_clicked(on_forward)
    buttons['Pause'].on_clicked(on_pause)
    buttons['Restart'].on_clicked(on_restart)

    # Add title to the figure
    fig.suptitle('Interactive Pathfinding Algorithm Visualizer', 
                fontsize=17, fontweight='bold', color='#1565C0', y=0.98)
    
    # Initial animation with smoother interval
    state['is_playing'] = True
    state['animation'] = animation.FuncAnimation(
        fig, update, fargs=(visited_order, path, method_name),
        frames=len(visited_order) + 15, interval=600, repeat=False, blit=False)

    plt.subplots_adjust(left=0.04, right=0.97, top=0.94, bottom=0.12, wspace=0.25, hspace=0.4)
    plt.show()

# ------------------------------
# SEARCH ALGORITHMS
# ------------------------------
def dfs(edges, start, goal):
    stack = [(start, [start], None)]
    visited = set()
    visited_order = []
    count = 0

    while stack:
        node, path, parent = stack.pop()
        count += 1
        visited_order.append((node, parent))

        if node in goal:
            return node, count, path, visited_order
        
        if node not in visited:
            visited.add(node)
            for neighbor, _ in sorted(edges.get(node, []), reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], node))

    return None, count, [], visited_order


def bfs(edges, start, goal):
    queue = deque([(start, [start], None)])
    visited = set()
    visited_order = []
    count = 0

    while queue:
        node, path, parent = queue.popleft()
        count += 1
        visited_order.append((node, parent))

        if node in goal:
            return node, count, path, visited_order
        
        if node not in visited:
            visited.add(node)
            for neighbor, _ in sorted(edges.get(node, [])):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], node))

    return None, count, [], visited_order


def gbfs(nodes, edges, start, goal):
    goal = goal[0]
    frontier = []
    heapq.heappush(frontier, (0, start, [start], None))
    visited = set()
    visited_order = []
    count = 0

    while frontier:
        _, node, path, parent = heapq.heappop(frontier)
        count += 1
        visited_order.append((node, parent))

        if node == goal:
            return node, count, path, visited_order

        if node not in visited:
            visited.add(node)
            for neighbor, _ in edges.get(node, []):
                if neighbor not in visited:
                    h = heuristic(neighbor, goal, nodes)
                    heapq.heappush(frontier, (h, neighbor, path + [neighbor], node))

    return None, count, [], visited_order


def astar(nodes, edges, start, goal):
    goal = goal[0]
    frontier = []
    heapq.heappush(frontier, (0, 0, start, [start], None))
    visited = {}
    visited_order = []
    count = 0

    while frontier:
        f, g, node, path, parent = heapq.heappop(frontier)
        count += 1
        visited_order.append((node, parent))

        if node == goal:
            return node, count, path, visited_order

        if node not in visited or g < visited[node]:
            visited[node] = g
            for neighbor, cost in sorted(edges.get(node, [])):
                g2 = g + cost
                f2 = g2 + heuristic(neighbor, goal, nodes)
                heapq.heappush(frontier, (f2, g2, neighbor, path + [neighbor], node))

    return None, count, [], visited_order

def ucs(edges, start, goal):
    print()

def custom_uninformed(edges, start, goal):
    # Wrapper for the CUS1 algorithm
    # Calls the Uniform Cost Search (UCS) implementation
    return ucs(edges, start, goal)

# ------------------------------
# CUSTOM INFORMED SEARCH (CUS2) – WEIGHTED A* SEARCH
# ------------------------------
# Description:
#   - Weighted A* is a variant of A* search that uses a weighted heuristic function:
#         f(n) = g(n) + w * h(n)
#     where:
#         g(n) = actual cost from start to current node
#         h(n) = estimated cost from current node to goal
#         w    = weight factor (>1) that biases the search toward the heuristic.
#   - When w = 1, it behaves like normal A*.
#   - When w > 1, it becomes greedier (faster but less optimal).
#
# Characteristics:
#   • Type: Informed
#   • Strategy: Balances actual and estimated cost based on the weight factor
#   • Optimal: Not guaranteed (for w > 1)
#   • Complete: Yes (for consistent heuristic)
#
# Implementation Details:
#   - Uses a priority queue ordered by the weighted f(n) value.
#   - Can trade accuracy for speed depending on the weight chosen.
#   - Ideal for large graphs where exact optimality is less important than faster performance.
def weighted_astar(nodes, edges, start, goal, weight=1.5):
    goal = goal[0]
    frontier = [(0, 0, start, [start], None)]
    visited = {}
    visited_order = []
    count = 0

    while frontier:
        f, g, node, path, parent = heapq.heappop(frontier)
        count += 1
        visited_order.append((node, parent))

        if node == goal:
            return node, count, path, visited_order

        if node not in visited or g < visited[node]:
            visited[node] = g
            for neighbor, cost in edges.get(node, []):
                g2 = g + cost
                f2 = g2 + weight * heuristic(neighbor, goal, nodes)
                heapq.heappush(frontier, (f2, g2, neighbor, path + [neighbor], node))

    return None, count, [], visited_order


def custom_informed(nodes, edges, start, goal):
    # Wrapper for the CUS2 algorithm
    # Calls the Weighted A* implementation with weight = 1.5
    return weighted_astar(nodes, edges, start, goal, weight=1.5)

# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename, method = sys.argv[1], sys.argv[2].upper()
    nodes, edges, origin, destinations = load_problem(filename)

    if method == "DFS":
        goal, count, path, visited_order = dfs(edges, origin, destinations)
    elif method == "BFS":
        goal, count, path, visited_order = bfs(edges, origin, destinations)
    elif method == "GBFS":
        goal, count, path, visited_order = gbfs(nodes, edges, origin, destinations)
    elif method in ("A*", "AS"):
        goal, count, path, visited_order = astar(nodes, edges, origin, destinations)
    elif method in ("UCS", "CUS1"):
        goal, count, path, visited_order = custom_uninformed(edges, origin, destinations)
    elif method in ("WA*", "CUS2"):
        goal, count, path, visited_order = custom_informed(nodes, edges, origin, destinations)
    else:
        print(f"Unknown method: {method}")
        print("Available methods: DFS, BFS, GBFS, A*, UCS/CUS1, WA*/CUS2")
        sys.exit(1)


    if goal:
        print_results(filename, method, goal, count, path)
        
        # Create full search tree from ALL graph nodes (not just visited)
        bst = create_bst_from_all_nodes(list(nodes.keys()))
        
        # Show both visualizations side-by-side
        print("\n" + "="*60)
        print("Building and visualizing Full Search Tree...")
        print("="*60)
        visualize_search(nodes, edges, visited_order, path, method, origin, destinations, bst=bst)
    else:
        print(f"{filename} {method}\nNo path found.")
# ------------------------------