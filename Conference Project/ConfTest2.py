from collections import deque
import heapq


GOAL = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0)
)



# Board helper functions

def find_blank(board):
    # Find the position of the blank tile (0)
    for r in range(3):
        for c in range(3): 
            if board[r][c] == 0: 
                return r, c


def swap(board, r1, c1, r2, c2): 
    board_list = [list(row) for row in board] # make copy that can be edited
    board_list[r1][c1], board_list[r2][c2] = board_list[r2][c2], board_list[r1][c1] # swap the two positions 
    return tuple(tuple(row) for row in board_list)   # back to tuple-of-tuples it can go into things


def successors(board):
    r, c = find_blank(board)  # generate all legal next moves by moving the blank
    moves = []
    directions = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]  # up, down, left, right
    for dr, dc, name in directions: # change in rows and cols 
        nr, nc = r + dr, c + dc # compute new position 

        if 0 <= nr < 3 and 0 <= nc < 3:  # bounds / only moves that stay inside the board
            new_board = swap(board, r, c, nr, nc)
            moves.append((new_board, name)) # new state, store board + move 

    return moves


# Manhattan distance 

def manhattan(board, goal=GOAL): # determines the heuristic value h(n)
    total = 0
    for r in range(3):
        for c in range(3):
            value = board[r][c] # loop through tiles 
            if value != 0:# ignore blank      
                for gr in range(3): # finds where  tile belongs go in the goal board
                    for gc in range(3):
                        if goal[gr][gc] == value:
                            total += abs(r - gr) + abs(c - gc) # abs of row-goalrow + abs col-goalcol (man dist)

    return total

# BFS solver

def bfs(start, goal=GOAL):
    queue = deque([start]) # FIFO 
    visited = {start} # no revisits 
    parent = {start: (None,None)} # prev state, move used 
    nodes_explored = 0 # number of configs expanded 

    while queue:
        current = queue.popleft()
        nodes_explored += 1 # work done 
        if current == goal:
            path = []
            state = current
            while state is not None:
                prev, move = parent[state]
                path.append((state, move))
                state = prev # walk backward 
            path.reverse() # start -> goal 
            return path, nodes_explored
        for nxt, move in successors(current): # all next states neighbors
            if nxt not in visited: # no repeats 
                visited.add(nxt)
                parent[nxt] = (current,move) # track 
                queue.append(nxt)# put in queue
    return None, nodes_explored


# A* solver
def astar(start, goal=GOAL):
    heap = [(manhattan(start, goal), 0, start)] # heap stores (priority, moves_so_far, board)
    parent = {start: (None, None)}
    cost_so_far = {start: 0} # g(n) exact cost 
    nodes_explored = 0

    while heap:
        priority, g, current = heapq.heappop(heap) # pick lowest f(n) total cost 
        nodes_explored +=1
        if current == goal:
            path = []
            state = current
            while state is not None:
                prev, move = parent[state]
                path.append((state, move))
                state = prev
            path.reverse() # same as bfs 
            return path, nodes_explored
        for nxt, move in successors(current):
            new_cost = g + 1 # compute g(n) 
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]: # only update if better path 
                cost_so_far[nxt] = new_cost
                parent[nxt] = (current, move)
                f = new_cost + manhattan(nxt, goal) # A* priority = g(n) + h(n)
                heapq.heappush(heap, (f, new_cost, nxt)) # add to queue 

    return None, nodes_explored



# Printing 
def print_board(board):
    for row in board:
        print(row)
    print()

def print_solution(path, label):
    if path is None:
        print(f"{label}: No solution found.\n")
        return

    print(f"{label}: Solution found in {len(path) - 1} moves.\n")
    
    for i, (board, move) in enumerate(path):
        if move is None:
            print(f"Step {i} (start):")
        else:
            print(f"Step {i} (move blank {move}):")

        print_board(board)
        print()


# Main
def main():
    puzzles = {

        "1": ("Easy", (
            (1, 2, 3),
            (4, 0, 6),
            (7, 5, 8)
        )),
        "2": ("Medium", (
            (1, 3, 6),
            (5, 0, 2),
            (4, 7, 8)
        )),
        "3": ("Hard", (
            (7, 1, 2),
            (4, 8, 5),
            (6, 3, 0)
        )),
        "4": ("Hardest", (
            (8, 6, 7),
            (2, 5, 4),
            (3, 0, 1)
        )),
         "5": ("One that I struggled with", (
            (4, 1, 2),
            (6, 8, 3),
            (5, 7, 0)
         ))
           
        
    }


    print("Choose a puzzle:")
    print("1. Easy")
    print("2. Medium")
    print("3. Hard")
    print("4. Hardest")
    print("5. One that I struggled with")
    choice = input("Enter 1, 2, 3, 4, or 5: ")

    if choice not in puzzles:
        print("Invalid choice, defaulting to Easy.")
        choice = "1"

    name, start = puzzles[choice]

    print(f"\nSelected: {name}")
    print("Start board:")
    print_board(start)

    bfs_path, bfs_nodes = bfs(start)
    astar_path, astar_nodes = astar(start)

    print_solution(bfs_path, "BFS")
    print_solution(astar_path, "A*")

    print("\nComparison:")
    print(f"BFS nodes expanded: {bfs_nodes}")
    print(f"A* nodes expanded: {astar_nodes}")

if __name__ == "__main__":
    main()