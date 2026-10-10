from collections import deque

# Goal board
GOAL = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0)
)

def find_blank(board):
    # Find where the 0 (blank) is
    for r in range(3):
        for c in range(3):
            if board[r][c] == 0:
                return r, c
    return None 

def swap(board, r1, c1, r2, c2):
    # Convert tuple board into list so we can modify it
    board_list = [list(row) for row in board]
    # Swap two positions
    board_list[r1][c1], board_list[r2][c2] = board_list[r2][c2], board_list[r1][c1]
    # Convert back into tuple-of-tuples so it can go in a set/dictionary
    return tuple(tuple(row) for row in board_list)

def successors(board):
    # Find blank location
    r, c = find_blank(board)
    next_boards = []
    # Possible directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = r + dr, c + dc

        # Only keep moves that stay inside the board
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_board = swap(board, r, c, nr, nc)
            next_boards.append(new_board)

    return next_boards

def bfs(start, goal=GOAL):
    # Queue for BFS
    q = deque([start])
    # Track visited boards so we do not repeat states
    visited = {start}
    # parent[state] = state that came before it
    parent = {start: None}
    while q:
        cur = q.popleft()
        # Check if solved
        if cur == goal:
            path = []
            s = cur
            while s is not None:
                path.append(s)
                s = parent[s]
            path.reverse()
            return path
        # Expand all legal next boards
        for nxt in successors(cur):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = cur
                q.append(nxt)
    return None

def print_board(board):
    for row in board:
        print(row)
    print()

def print_solution(path):
    for i, board in enumerate(path):
        print(f"Step {i}:")
        print_board(board)

def main():
    # A simple solvable start board
    start = (
        (4, 1, 3),
        (7, 2, 5),
        (0, 8, 6),
    )

    path = bfs(start)

    if path is None:
        print("No solution found.")
    else:
        print(f"Solution found in {len(path) - 1} moves.\n")
        print_solution(path)

if __name__ == "__main__":
    main()

