from collections import deque  
import heapq                  


# board helper 

def find_blank(board):
    for r in range(9): 
        for c in range(9): 
            if board[r][c] == 0:  # if 0 then blank cell
                return r, c  

    return None #(board is full)



def is_valid(board, row, col, num): #  sudoku rules / constraints 
    for c in range(9):  # row
        if board[row][c] == num:  # check if num in row if so then invalid
            return False          

    for r in range(9):  # column
        if board[r][col] == num:  # check if num exists in column
            return False

    # Find topleft corner of the 3x3 box
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    # loop through the  box
    for r in range(start_row, start_row + 3): # loop through each box 
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:  # see if num already exists in box
                return False

    
    return True # valid 


def successors(board):
    moves = []
    blank = find_blank(board) # choose first blank 
    if blank is None:
        return moves 
    
    r, c = blank # unpack row column blank 
    for num in range(1, 10): # try 1-9 
        if is_valid(board, r, c, num): # check if valid, if so keep and make new board 
            board_list = [list(row) for row in board] # turn to list 
            board_list[r][c] = num # try num 
            new_board = tuple(tuple(row) for row in board_list) # back to tuple 
            moves.append((new_board, f"place {num} at ({r}, {c})")) # if valid add to move list 
    return moves


# heuristic 
def count_blanks(board):
    total = 0  # blanks

    for r in range(9): # loop board 
        for c in range(9):
            if board[r][c] == 0:  # if blank
                total += 1        

    return total  # number of blanks

def is_goal(board):
    return find_blank(board) is None # goal reached when no blanks left


# A* SEARCH
def astar(start):
    heap = [(count_blanks(start), 0, start)] # start pqueue , cost 0 
    parent = {start: (None, None)} # store prev board, move 
    cost_so_far = {start: 0} #g(n)
    nodes_explored = 0

    while heap:
        priority, g, current = heapq.heappop(heap) # take smallest f(n) 
        nodes_explored += 1
        if is_goal(current): #solved? retrace steps 
            path = []
            state = current # cur board lowest f(n)
            while state is not None: #until no parent 
                prev, move = parent[state] 
                path.append((state, move)) # add to path
                state = prev 
            path.reverse()
            return path, nodes_explored

        for nxt, move in successors(current): # if not solved explore next legal states 
            new_cost = g + 1 
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]: # only update if better path  
                cost_so_far[nxt] = new_cost
                parent[nxt] = (current, move) # what move got us to best known cost 
                f = new_cost + count_blanks(nxt) # f(n) = g(n) + h(n)
                heapq.heappush(heap, (f, new_cost, nxt))
                # become new candidate 
    return None, nodes_explored



def print_board(board):
    for r in range(9):
        if r % 3 == 0 and r != 0: 
            print("-" * 21)

        for c in range(9):
            if c % 3 == 0 and c != 0:
                print("|", end=" ")
            print(board[r][c], end=" ")

        print()
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
            print(f"Step {i} ({move}):")

        print_board(board)


def main():
    puzzles = {
        "1": ("Easy", (
            (5, 3, 0, 0, 7, 0, 0, 0, 0),
            (6, 0, 0, 1, 9, 5, 0, 0, 0),
            (0, 9, 8, 0, 0, 0, 0, 6, 0),
            (8, 0, 0, 0, 6, 0, 0, 0, 3),
            (4, 0, 0, 8, 0, 3, 0, 0, 1),
            (7, 0, 0, 0, 2, 0, 0, 0, 6),
            (0, 6, 0, 0, 0, 0, 2, 8, 0),
            (0, 0, 0, 4, 1, 9, 0, 0, 5),
            (0, 0, 0, 0, 8, 0, 0, 7, 9)
        )),
        
        "2": ("Medium", (
            (0, 0, 0, 2, 6, 0, 7, 0, 1),
            (6, 8, 0, 0, 7, 0, 0, 9, 0),
            (1, 9, 0, 0, 0, 4, 5, 0, 0),
            (8, 2, 0, 1, 0, 0, 0, 4, 0),
            (0, 0, 4, 6, 0, 2, 9, 0, 0),
            (0, 5, 0, 0, 0, 3, 0, 2, 8),
            (0, 0, 9, 3, 0, 0, 0, 7, 4),
            (0, 4, 0, 0, 5, 0, 0, 3, 6),
            (7, 0, 3, 0, 1, 8, 0, 0, 0)
        )),

        "3": ("Hard", (
            (0, 0, 0, 0, 0, 0, 2, 0, 0),
            (0, 8, 0, 0, 0, 7, 0, 9, 0),
            (6, 0, 2, 0, 0, 0, 5, 0, 0),
            (0, 7, 0, 0, 6, 0, 0, 0, 0),
            (0, 0, 0, 9, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 2, 0, 0, 4, 0),
            (0, 0, 5, 0, 0, 0, 6, 0, 3),
            (0, 9, 0, 4, 0, 0, 0, 7, 0),
            (0, 0, 6, 0, 0, 0, 0, 0, 0)
        )),

        "4": ("Very Hard", (
            (8, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 3, 6, 0, 0, 0, 0, 0),
            (0, 7, 0, 0, 9, 0, 2, 0, 0),
            (0, 5, 0, 0, 0, 7, 0, 0, 0),
            (0, 0, 0, 0, 4, 5, 7, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 3, 0),
            (0, 0, 1, 0, 0, 0, 0, 6, 8),
            (0, 0, 8, 5, 0, 0, 0, 1, 0),
            (0, 9, 0, 0, 0, 0, 4, 0, 0)
        )),

        "5": ("One I struggled with", (
            (4, 0, 0, 0, 5, 0, 0, 0, 0),
            (5, 0, 1, 0, 8, 0, 0, 0, 0),
            (0, 0, 0, 7, 9, 6, 0, 0, 0),
            (0, 0, 7, 0, 0, 0, 0, 0, 0),
            (0, 0, 9, 0, 0, 8, 6, 0, 0),
            (0, 5, 0, 0, 0, 0, 9, 0, 7),
            (0, 0, 0, 0, 0, 3, 0, 0, 5),
            (0, 0, 0, 6, 0, 0, 0, 8, 0),
            (2, 3, 0, 0, 0, 0, 0, 1, 0)
        ))
    }

    print("Choose a Sudoku puzzle:")
    print("1. Easy")
    print("2. Medium")
    print("3. Hard")
    print("4. Very Hard")
    print("5. One I struggled with")
    choice = input("Enter 1, 2, 3, 4, or 5: ")

    if choice not in puzzles:
        print("Invalid choice, defaulting to Easy.")
        choice = "1"

    name, start = puzzles[choice]

    print(f"\nSelected: {name}")
    print("Start board:")
    print_board(start)

    astar_path, astar_nodes = astar(start)

    print_solution(astar_path, "A*")
    print(f"A* nodes expanded: {astar_nodes}")


if __name__ == "__main__":
    main()