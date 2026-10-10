from collections import deque

def is_valid(state, M, C):
    mL, cL, boatL = state 
    mR, cR = M - mL, C - cL #compute how many on right 
    # bounds
    if mL < 0 or mL > M or cL < 0 or cL > C:
        return False
    # missionaries cannot be outnumbered on either bank 
    if mL > 0 and cL > mL:
        return False
    if mR > 0 and cR > mR:
        return False

    return True

def successors(state, M, C):
    mL, cL, boatL = state
    # possible boat moves with capacity 2 mish first, then canibal
    moves = [(2,0), (0,2), (1,1), (1,0), (0,1)]
    next_states = []
    for m_move, c_move in moves:
        if boatL: 
            ns = (mL - m_move, cL - c_move, False) # left to right
        else:      
            ns = (mL + m_move, cL + c_move, True) # right to left
        if is_valid(ns, M, C): #only keep if valid 
            next_states.append((ns, (m_move, c_move)))

    return next_states

def bfs(M=3, C=3):  
    start = (M, C, True)   # everything on left
    goal  = (0, 0, False)  # everyone on right 
    q = deque([start]) #bfs queue 
    visited = {start} # keep track, prevent revisiting 
    parent = {start: None}     # keep track of path 
    action = {start: None}     # track move used to get here 
    while q:
        cur = q.popleft()
        if cur[0] == goal[0] and cur[1] == goal[1]:
            path = [] # rebuild path
            s = cur
            while s is not None: 
                path.append(s)
                s = parent[s]
            path.reverse() # goes start to goal 
            return path, action
        for ns, move in successors(cur, M, C): # expand neighbors 
            if ns not in visited: # if new state, add to bfs 
                visited.add(ns) # visited
                parent[ns] = cur # how we got there 
                action[ns] = move # track move 
                q.append(ns) # put in queue 
    return None, None # if no solution exists 



def print_solution(path, action, M, C):
    for i, state in enumerate(path):
        mL, cL, boatL = state
        mR, cR = M - mL, C - cL
        side = "L" if boatL else "R" # boat loc as letter 
        if i == 0:
            print(f"{i}: START   Left({mL},{cL})  Right({mR},{cR})  Boat:{side}")
        else:
            move = action[state]
            print(f"{i}: move {move}  Left({mL},{cL})  Right({mR},{cR})  Boat:{side}")

def main():
    M, C = 3, 3
    path, action = bfs(M, C)

    if path is None:
        print("No solution")
    else:
        print_solution(path, action, M, C)