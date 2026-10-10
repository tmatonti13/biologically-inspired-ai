import heapq

# Goal board
GOAL = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0)
)

def find_blank(board):
    for r in range(3):
        for c in range(3):
            if board[r][c] == 0:
                return r, c

def swap(board, r1, c1, r2, c2):
    board_list = [list(row) for row in board]
    board_list[r1][c1], board_list[r2][c2] = board_list[r2][c2], board_list[r1][c1]
    return tuple(tuple(row) for row in board_list)

def successors(board):
    r, c = find_blank(board)
    next_boards = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_board = swap(board, r, c, nr, nc)
            next_boards.append(new_board)

    return next_boards

def misplaced_tiles(board, goal=GOAL):
    count = 0
    for r in range(3):
        for c in range(3):
            if board[r][c] != 0 and board[r][c] != goal[r][c]:
                count += 1
    return count

def greedy_best_first(start, goal=GOAL):
    # priority queue entries: (heuristic, counter, board)
    frontier = []
    counter = 0
    heapq.heappush(frontier, (misplaced_tiles(start, goal), counter, start))

    visited = {start}
    parent = {start: None}

    while frontier:
        h, _, cur = heapq.heappop(frontier)

        if cur == goal:
            path = []
            s = cur
            while s is not None:
                path.append(s)
                s = parent[s]
            path.reverse()
            return path

        for nxt in successors(cur):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = cur
                counter += 1
                heapq.heappush(frontier, (misplaced_tiles(nxt, goal), counter, nxt))

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
    start = (
        (1, 2, 3),
        (4, 0, 6),
        (7, 5, 8)
    )

    path = greedy_best_first(start)

    if path is None:
        print("No solution found.")
    else:
        print(f"Solution found in {len(path) - 1} moves.\n")
        print_solution(path)

if __name__ == "__main__":
    main()