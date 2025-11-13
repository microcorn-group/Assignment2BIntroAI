
# Intro AI — Assignment 2a: Path Finding

This repository contains a path-finding assignment implemented in Python. The code demonstrates several uninformed and informed search algorithms (DFS, BFS, Greedy Best-First Search, A*) on a graph with an **interactive step-by-step visualization** and a Binary Search Tree (BST) showing the search order.

## Contents

- `search.py` — Main program implementing:
	- Depth-First Search (DFS)
	- Breadth-First Search (BFS)
	- Greedy Best-First Search (GBFS)
	- A* Search (A*)
	- Two custom method aliases (`CUS1`, `CUS2`) that map to DFS and A* respectively
	- **🎮 Interactive UI with step-by-step controls** (Back, Pause/Resume, Forward, Restart)
	- **Dual visualization** - Graph on left, Search Tree (BST) on right
	- **Red exploration path** showing algorithm's search progression
	- **Green final path** overlaid on exploration
	- **Video-like progress bar** with percentage display
	- Utilities to load a problem file, print results, and display interactive matplotlib visualization
- `search.py` — Contains BST visualization with Reingold-Tilford layout algorithm
- `bst_visualizer.py` — Binary Search Tree visualization with optimized node positioning
- `PathFinder-test.txt` — Example problem file with nodes, edges, origin and destinations
- `com_path.txt` — Complex test case with 18 nodes and multiple routing options
- `UI_GUIDE.md` — Comprehensive guide for using the interactive controls

## Problem file format

The problem files are plain text and contain four sections: `Nodes:`, `Edges:`, `Origin:`, and `Destinations:`. Sections are case-sensitive and should appear exactly as shown. Example (from `PathFinder-test.txt`):

Nodes:
1: (4,1)
2: (2,2)
3: (4,4)
4: (6,3)
5: (5,6)
6: (7,5)

Edges:
(2,1): 4
(3,1): 5
(1,3): 5
... (each edge is written as `(from,to): cost`)

Origin:
2

Destinations:
5; 4

Notes:
- Node coordinates are used by the heuristic (Euclidean distance) and for plotting.
- Multiple destinations can be listed separated by semicolons (`;`). The current informed algorithms (GBFS, A*) assume a single goal and will use the first destination listed for the heuristic calculation.

## Requirements

- Python 3.8+ (should work with Python 3.7+, but 3.8+ is recommended)

All Python dependencies are listed in `requirements.txt`. It's recommended to use a virtual environment. To install the dependencies, run:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the `search.py` script with two arguments: the problem filename and the search method.

Supported methods (case-insensitive):
- `DFS` — Depth-First Search (unweighted)
- `BFS` — Breadth-First Search (unweighted)
- `GBFS` — Greedy Best-First Search (weighted, uses heuristic only)
- `A*` or `AS` — A* Search (weighted, uses f = g + h)
- `CUS1` — Custom uninformed search (Uniform Cost Search - UCS)
- `CUS2` — Custom informed search (Weighted A* with w=1.5)

Example (bash/PowerShell):

```bash
python search.py PathFinder-test.txt DFS
```

Expected output (example):

```
PathFinder-test.txt DFS
5 10
2 -> 3 -> 5
```

After the textual output the program will open an **interactive matplotlib window** with comprehensive visualization features.

## 🎮 Interactive Visualization Features

The visualization window displays:

### Layout
- **Left Side**: Search graph with nodes, edges, and costs
- **Right Side**: Binary Search Tree showing algorithm's exploration order
- **Top**: Algorithm info box and legend
- **Bottom**: Video-like progress bar with percentage, and control buttons

### Visual Elements
- **Color-coded nodes**:
  - Green: Start node
  - Red: Goal node(s)
  - Orange: Explored nodes
  - Gray: Unvisited nodes
- **Red exploration path**: Shows the algorithm's search progression through parent-child relationships
- **Green final path**: Overlaid on top showing the optimal solution
- **Visit order badges**: Numbers (#1, #2, etc.) on each explored node
- **Progress bar**: Blue while searching, green when complete, with percentage display

### Control Buttons
Located at the bottom of the window:
- **[Back]** - Step backward through the search visualization frame-by-frame
- **[Pause/Resume]** - Pause the animation or resume from where it stopped
- **[Forward]** - Step forward through the search visualization frame-by-frame
- **[Restart]** - Restart the animation from the beginning with automatic playback

### Interactive Workflow
1. **Automatic Playback**: Animation starts automatically showing each step of the algorithm
2. **Manual Stepping**: Click Back/Forward to manually step through the search at your own pace
3. **Pause & Inspect**: Click Pause to freeze at any point and examine the exploration state
4. **Restart & Compare**: Click Restart to replay or run with a different algorithm

## Algorithm Details

### Search Algorithms Implemented
- **DFS (Depth-First Search)**: Uninformed, explores deeply before backtracking
- **BFS (Breadth-First Search)**: Uninformed, explores breadth-first ensuring shortest path in unweighted graphs
- **GBFS (Greedy Best-First Search)**: Informed, uses heuristic to greedily select most promising node
- **A* Search**: Informed, combines actual cost and heuristic (f = g + h) for optimal pathfinding

### Heuristic Function
Euclidean distance is used for informed algorithms:
```
heuristic(node_a, node_b) = sqrt((x_a - x_b)² + (y_a - y_b)²)
```

### Search Tree Visualization
The BST on the right shows:
- **Node numbering**: Visit order (#1, #2, etc.) when explored
- **Tree structure**: Parent-child relationships from algorithm's exploration
- **Color highlighting**: Active/inactive nodes to show progression
- **Edge connections**: Shows how the algorithm built its search tree

## Screenshots

The interactive visualizer displays a comprehensive side-by-side view:
![Screenshot1](images/Screenshot%20From%202025-10-28%2007-32-15.png)

![Screenshot2](images/Screenshot%20From%202025-10-28%2008-12-16.png)

![Screenshot3](images/Screenshot%20From%202025-10-28%2008-13-37.png)

![Screenshot4](images/screenshot.gif)

Shows the final result with the search complete. The green final path is overlaid on the red exploration path, the progress bar is at 100% (green), and all visited nodes are highlighted with their visit order numbers.

### Main Visualization Window Features
- **Left Panel**: Search graph showing nodes, edges, costs, and the exploration path
- **Right Panel**: Binary Search Tree showing the order of node exploration
- **Top**: Algorithm info box with current status and statistics
- **Middle**: Legend showing color meanings and path types
- **Bottom**: Video-like progress bar and control buttons

### Key Visual Indicators
- **Start node** (Green): Where the search begins
- **Goal nodes** (Red): Target destinations
- **Explored nodes** (Orange): Nodes visited during search with visit order numbers
- **Unvisited nodes** (Gray): Nodes not yet explored
- **Exploration path** (Red lines): Shows the search tree as it expands
- **Final path** (Green lines): Optimal solution overlaid on exploration

### Progress Tracking
- **Video-like progress bar**: Shows search progress percentage in real-time
- **Percentage display**: Current progress from 0% to 100%
- **Color transition**: Blue while searching, turns green upon completion
- **Info box**: Displays algorithm name, nodes explored, current node, and status

## Notes and caveats


- The input parser expects the exact section headers `Nodes:`, `Edges:`, `Origin:`, and `Destinations:` as shown. Empty lines are ignored.
- The informed search implementations (GBFS and A*) currently use only the first listed destination when computing heuristic distances. If you need multi-goal heuristics, the code will need to be extended.
- Edge definitions are directional as listed in the file (the example contains edges in both directions where appropriate). Make sure to list both (a,b) and (b,a) if you want a bidirectional connection with potentially different costs.
