def bfs(self):
    start_time = time.time()
    queue = deque([(self, [])])
    visited = set()
    nodes_explored = 0

    while queue:
        state, path = queue.popleft()
        nodes_explored += 1

        if state.is_solved():
            end_time = time.time()
            for move in path:
                print(move)
            state.display_state()
            print(f"Number of Nodes Visited: {nodes_explored}")
            print(f"Total Search Time: {(end_time - start_time) * 1000:.2f} ms")
            print(f"Solution Path Length: {len(path)}")

            return path

        visited.add(tuple(map(tuple, state.board)))

        for move in state.get_available_moves():
            piece, direction = move
            new_state = state.apply_move(piece, direction)
            new_state = new_state.normalize()
            if tuple(map(tuple, new_state.board)) not in visited:
                queue.append((new_state, path + [move]))
    print("no path found")
    return None


def dfs(self):
    start_time = time.time()
    stack = [(self, [])]
    visited = set()
    nodes_explored = 0

    while stack:
        state, path = stack.pop()
        nodes_explored += 1

        if state.is_solved():
            end_time = time.time()
            for move in path:
                print(move)
            state.display_state()
            print(f"Number of Nodes Visited: {nodes_explored}")
            print(f"Total Search Time: {(end_time - start_time) * 1000:.2f} ms")
            print(f"Solution Path Length: {len(path)}")

            return path

        visited.add(tuple(map(tuple, state.board)))

        for move in reversed(state.get_available_moves()):
            piece, direction = move
            new_state = state.apply_move(piece, direction)
            new_state = new_state.normalize()
            if tuple(map(tuple, new_state.board)) not in visited:
                stack.append((new_state, path + [move]))
    return None

def astar(self):
    start_time = time.time()
    priority_queue = []
    counter = itertools.count()  # Unique tie-breaker

    initial_state = self.normalize()  # Normalize before starting
    heapq.heappush(priority_queue, (0, 0, next(counter), initial_state, []))

    visited = set()
    nodes_explored = 0

    while priority_queue:
        _, g, _, state, path = heapq.heappop(priority_queue)  # Stop when dequeued
        nodes_explored += 1

        if state.is_solved():  # Stop when goal state is dequeued
            end_time = time.time()
            for move in path:
                print(move)
            state.display_state()
            print(f"Number of Nodes Visited: {nodes_explored}")
            print(f"Total Search Time: {(end_time - start_time) * 1000:.2f} ms")
            print(f"Solution Path Length: {len(path)}")
            return path

        normalized_state = state.normalize()  # Normalize before storing
        visited.add(tuple(map(tuple, normalized_state.board)))

        available_moves = state.get_available_moves()
        for move in available_moves:
            piece, direction = move
            new_state = state.apply_move(piece, direction)
            new_state = new_state.normalize()  # Normalize new state before storing

            if tuple(map(tuple, new_state.board)) not in visited:
                g_new = g + 1  # Cost so far
                h_new = new_state.heuristic()  # Estimated cost to goal
                f_new = g_new + h_new  # Total cost function f(n)

                heapq.heappush(priority_queue, (f_new, g_new, next(counter), new_state, path + [move]))

    print("No solution found.")  # If A* exhausts all possibilities
    return None