import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import math
import time # Import time for animation pause

# --- 1. BST Data Structure (Unchanged/Refined) ---

class BSTNode:
    """Binary Search Tree Node"""
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.visit_order = None
        self.is_visited = False  # Track if node has been visited/highlighted

class BST:
    """Binary Search Tree implementation"""
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        """Insert a value into the BST"""
        if self.root is None:
            self.root = BSTNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        """Recursive helper for insert"""
        # Ensure only unique values are added for a strict BST visualization
        if value == node.value:
            return 
        elif value < node.value:
            if node.left is None:
                node.left = BSTNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = BSTNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    # Traversal methods (kept for completeness, but not strictly used by the visualizer yet)
    # ... (inorder_traversal, preorder_traversal, postorder_traversal, etc.) ...
    
    def set_visit_order(self, visited_list_values):
        """Set visit order numbers for each node"""
        # Map raw values to their visit order (1-based index)
        visited_dict = {value: order + 1 for order, value in enumerate(visited_list_values)}
        self._set_visit_recursive(self.root, visited_dict)
    
    def _set_visit_recursive(self, node, visited_dict):
        """Recursive helper to set visit order"""
        if node is None:
            return
        if node.value in visited_dict:
            node.visit_order = visited_dict[node.value]
        self._set_visit_recursive(node.left, visited_dict)
        self._set_visit_recursive(node.right, visited_dict)
    
    def get_all_nodes(self):
        """Get all nodes in a list (for tracking)"""
        result = []
        self._collect_nodes(self.root, result)
        return result
    
    def _collect_nodes(self, node, result):
        """Recursively collect all nodes"""
        if node is None:
            return
        result.append(node)
        self._collect_nodes(node.left, result)
        self._collect_nodes(node.right, result)

# --- 2. Tree Construction Functions (Refined) ---

def create_bst_from_all_nodes(all_nodes):
    """
    Create a (relatively) balanced Binary Search Tree from a list of all node values.
    """
    if not all_nodes:
        return None
    
    # Sort nodes to create a balanced BST
    sorted_nodes = sorted(list(set(all_nodes))) # ensure unique and sorted
    
    def build_balanced_bst(node_list):
        if not node_list:
            return None
        mid = len(node_list) // 2
        node = BSTNode(node_list[mid])
        node.left = build_balanced_bst(node_list[:mid])
        node.right = build_balanced_bst(node_list[mid+1:])
        return node
    
    bst = BST()
    bst.root = build_balanced_bst(sorted_nodes)
    return bst

# --- 3. Layout and Drawing Functions (Aesthetics Refined) ---

def calculate_tree_layout(node, x=0, y=0, dx=150):
    """
    Calculate positions for tree nodes using Reingold-Tilford algorithm variant.
    This prevents edge crossings by calculating subtree widths.
    """
    if node is None:
        return []
    
    dy = 120  # Vertical spacing
    
    positions = [(node.value, x, y)]
    
    # Get widths of left and right subtrees to position children correctly
    left_positions = []
    right_positions = []
    
    if node.left is not None:
        # Calculate the width needed for the left subtree
        left_width = get_subtree_width(node.left)
        left_x = x - dx - (left_width * dx) / 2
        left_positions = calculate_tree_layout(node.left, left_x, y - dy, dx * 0.8)
        positions.extend(left_positions)
    
    if node.right is not None:
        # Calculate the width needed for the right subtree
        right_width = get_subtree_width(node.right)
        right_x = x + dx + (right_width * dx) / 2
        right_positions = calculate_tree_layout(node.right, right_x, y - dy, dx * 0.8)
        positions.extend(right_positions)
    
    return positions


def get_subtree_width(node):
    """
    Calculate the width (number of leaf nodes) of a subtree.
    Used to position nodes to avoid edge crossings.
    """
    if node is None:
        return 0
    
    if node.left is None and node.right is None:
        return 1  # Leaf node
    
    left_width = get_subtree_width(node.left)
    right_width = get_subtree_width(node.right)
    
    return left_width + right_width if (left_width + right_width) > 0 else 1


def draw_tree_edges(ax, node, positions_dict, color='#AABBC3', linewidth=2.0):
    """Draw edges between tree nodes"""
    if node is None:
        return
    
    x, y = positions_dict[node.value]

    if node.left is not None:
        x_child, y_child = positions_dict[node.left.value]
        # Draw a line from parent to child
        ax.plot([x, x_child], [y, y_child], '-', linewidth=linewidth,
                color=color, zorder=1, alpha=0.7)
        draw_tree_edges(ax, node.left, positions_dict, color, linewidth)
    
    if node.right is not None:
        x_child, y_child = positions_dict[node.right.value]
        # Draw a line from parent to child
        ax.plot([x, x_child], [y, y_child], '-', linewidth=linewidth,
                color=color, zorder=1, alpha=0.7)
        draw_tree_edges(ax, node.right, positions_dict, color, linewidth)


def setup_bst_visualization(ax_bst, bst, bst_positions_dict):
    """
    Draw the complete search tree structure (static setup).
    Returns a dict of node_circles, node_texts, and node_badges for animation.
    """
    if bst.root is None:
        return {}, {}, {}
    
    node_circles = {}
    node_texts = {}
    node_badges = {}
    
    # 1. Draw Edges (Background/Inactive)
    if bst.root.value in bst_positions_dict:
        draw_tree_edges(ax_bst, bst.root, bst_positions_dict)
    
    # 2. Draw Nodes (Inactive/Gray at first)
    for value, (x, y) in bst_positions_dict.items():
        # Node Circle
        circle = patches.Circle((x, y), 25,
                               facecolor='#F5F5F5',      # Very light gray (inactive)
                               edgecolor='#AABBC3',      # Soft gray border
                               linewidth=2.0, zorder=5)
        ax_bst.add_patch(circle)
        node_circles[value] = circle
        
        # Node Value Text
        text = ax_bst.text(x, y, str(value), fontsize=14, ha='center', va='center',
                          fontweight='bold', color='#455A64', zorder=6) # Darker gray text
        node_texts[value] = text
        
        # Visit Order Badge (Hidden initially)
        badge = ax_bst.text(x + 18, y + 18, "", fontsize=10, ha='center', va='center',
                           fontweight='heavy', color='#FFFFFF',
                           bbox=dict(boxstyle='circle,pad=0.2',
                                    facecolor='#03A9F4', edgecolor='#0288D1',
                                    alpha=0.0, linewidth=0.0), zorder=7) # Light Blue, fully transparent
        node_badges[value] = badge
    
    return node_circles, node_texts, node_badges


def highlight_node(node_circles, node_texts, node_badges, node_value, visit_num):
    """
    Highlight a node when visited.
    Changes from gray (inactive) to a distinct color (active).
    """
    if node_value in node_circles:
        # Active state: Distinct color (e.g., Orange/Amber)
        node_circles[node_value].set_facecolor('#FFB300') # Amber
        node_circles[node_value].set_edgecolor('#FF8F00') # Darker Amber
        node_circles[node_value].set_linewidth(3.5)
        
        # White text for contrast
        node_texts[node_value].set_color('#FFFFFF')
        
        # Show visit badge (Blue for order)
        badge = node_badges[node_value]
        badge.set_text(f"{visit_num}")
        bbox = badge.get_bbox_patch()
        if bbox:
            bbox.set_facecolor('#03A9F4') # Blue badge
            bbox.set_edgecolor('#0288D1')
            bbox.set_alpha(1.0)
            bbox.set_linewidth(1.5)

# --- 4. Animation/Run Function (New) ---

def animate_traversal(bst, traversal_order_values, node_circles, node_texts, node_badges, fig):
    """
    Iterates through the traversal order and highlights nodes one by one.
    """
    print(f"Starting traversal visualization for: {traversal_order_values}")
    
    # Find all nodes for quick lookup
    all_nodes_map = {n.value: n for n in bst.get_all_nodes()}
    
    # Loop through the order
    for step, node_value in enumerate(traversal_order_values):
        visit_num = step + 1
        
        # 1. Update the node's appearance
        highlight_node(node_circles, node_texts, node_badges, node_value, visit_num)
        
        # 2. Update the plot to show the change
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # 3. Pause for visibility
        time.sleep(0.7) # Adjust speed here
    
    print("Traversal complete.")


def visualize_bst_traversal(all_node_values, traversal_order_values, traversal_name):
    """
    Main function to set up and run the visualization.
    """
    # 1. Build the Balanced BST
    bst = create_bst_from_all_nodes(all_node_values)
    if bst is None:
        print("Cannot visualize an empty tree.")
        return

    # 2. Calculate Layout
    # Use a large initial factor for wider spacing at the top
    positions_list = calculate_tree_layout(bst.root, dx_factor=1.2) 
    bst_positions_dict = {value: (x, y) for value, x, y in positions_list}
    
    # Find max/min coordinates for setting plot limits
    if not bst_positions_dict:
         print("No nodes to display.")
         return
         
    all_x = [x for v, (x, y) in bst_positions_dict.items()]
    all_y = [y for v, (x, y) in bst_positions_dict.items()]

    # 3. Setup Matplotlib Figure
    plt.ion() # Turn on interactive mode for live updates
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set title and remove axes/frame for a cleaner look
    ax.set_title(f'BST Traversal Visualization: {traversal_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set limits with padding (50 units of padding on each side)
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    ax.set_xlim(x_min - 50, x_max + 50)
    ax.set_ylim(y_min - 50, y_max + 50)
    ax.invert_yaxis() # Tree drawing usually looks better with y=0 at the top

    # 4. Draw Static Structure
    node_circles, node_texts, node_badges = setup_bst_visualization(ax, bst, bst_positions_dict)

    # 5. Run Animation
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1) # Initial pause
    
    animate_traversal(bst, traversal_order_values, node_circles, node_texts, node_badges, fig)
    
    # 6. Final State
    # Keep the plot open until closed by user
    plt.ioff()
    plt.show()

# --- 5. Example Usage ---

if __name__ == '__main__':
    # 1. Define the complete set of nodes that will be in the tree
    # The visualization will build a balanced BST from this list.
    ALL_NODES = [50, 30, 70, 20, 40, 60, 80, 15, 25, 35, 45, 55, 65, 75, 85]

    # 2. Define a traversal order (e.g., In-order traversal of the balanced BST)
    # The balanced BST from ALL_NODES will look like: 
    #      50
    #    /    \
    #  30      70
    # / \    / \
    # 20 40  60 80
    
    # In-order Traversal of the above structure:
    INORDER_TRAVERSAL = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    
    # Pre-order Traversal of the above structure:
    PREORDER_TRAVERSAL = [50, 30, 20, 15, 25, 40, 35, 45, 70, 60, 55, 65, 80, 75, 85]

    print("--- Running In-order Traversal Visualization ---")
    visualize_bst_traversal(
        all_node_values=ALL_NODES,
        traversal_order_values=INORDER_TRAVERSAL,
        traversal_name="In-order"
    )