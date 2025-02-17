import heapq
import itertools
import random
import sys
import time
from collections import deque


class SlidingBrickPuzzle:
    def __init__(self, width, height, board):
        self.width = width
        self.height = height
        self.board = board

    @classmethod
    def load_state(cls, filename):
        try:
            with open(filename, "r") as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]

            if not lines or len(lines) < 2:
                raise ValueError("File is empty or incorrectly formatted.")

            width, height = map(int, lines[0].strip().rstrip(",").split(","))
            board = []
            for line in lines[1:]:
                row_values = [val for val in line.strip().rstrip(",").split(",") if val.strip() != ""]
                board.append(list(map(int, row_values)))

            return cls(width, height, board)

        except Exception as e:
            print(f"Error loading state: {e}")
            sys.exit(1)

    def display_state(self):
        print(f"{self.width},{self.height},")
        for row in self.board:
            print(" " + ",".join(map(str, row)) + ",")

    def is_solved(self):
        for row in self.board:
            if -1 in row:
                return False
        return True

    def clone(self):
        new_board = [row[:] for row in self.board]
        return SlidingBrickPuzzle(self.width, self.height, new_board)

    def get_piece_positions(self, piece):
        positions = []
        for r in range(self.height):
            for c in range(self.width):
                if self.board[r][c] == piece:
                    positions.append((r, c))
        return positions

    def get_available_moves(self):
        moves = []
        checked_pieces = set()

        for r in range(self.height):
            for c in range(self.width):
                piece = self.board[r][c]

                if piece in {0, 1, -1} or piece in checked_pieces:
                    continue

                checked_pieces.add(piece)
                positions = self.get_piece_positions(piece)

                if self.can_move(positions, "up"):
                    moves.append((piece, "up"))
                if self.can_move(positions, "down"):
                    moves.append((piece, "down"))
                if self.can_move(positions, "left"):
                    moves.append((piece, "left"))
                if self.can_move(positions, "right"):
                    moves.append((piece, "right"))

        return moves

    def can_move(self, positions, direction):
        piece_value = self.board[positions[0][0]][positions[0][1]]

        allowed_spaces = {0} if piece_value != 2 else {0, -1}

        if direction == "up":
            top_edge = min(r for r, c in positions)
            return all(
                r > 0 and self.board[r - 1][c] in allowed_spaces
                for r, c in positions if r == top_edge
            )

        if direction == "down":
            bottom_edge = max(r for r, c in positions)
            return all(
                r < self.height - 1 and self.board[r + 1][c] in allowed_spaces
                for r, c in positions if r == bottom_edge
            )

        if direction == "left":
            left_edge = min(c for r, c in positions)
            return all(
                c > 0 and self.board[r][c - 1] in allowed_spaces
                for r, c in positions if c == left_edge
            )

        if direction == "right":
            right_edge = max(c for r, c in positions)
            return all(
                c < self.width - 1 and self.board[r][c + 1] in allowed_spaces
                for r, c in positions if c == right_edge
            )

        return False

    def apply_move(self, piece, direction):
        new_puzzle = self.clone()
        positions = new_puzzle.get_piece_positions(piece)

        new_positions = [(r - 1, c) if direction == "up" else
                         (r + 1, c) if direction == "down" else
                         (r, c - 1) if direction == "left" else
                         (r, c + 1) for r, c in positions]

        for r, c in positions:
            new_puzzle.board[r][c] = 0

        for r, c in new_positions:
            new_puzzle.board[r][c] = piece

        return new_puzzle

    def compare_states(self, other):
        if self.width != other.width or self.height != other.height:
            return False

        for r in range(self.height):
            for c in range(self.width):
                if self.board[r][c] != other.board[r][c]:
                    return False

        return True

    # def normalize(self):
    #     board = self.board
    #     piece_mapping = {}
    #     next_piece_number = 3
    #
    #     for r in range(self.height):
    #         for c in range(self.width):
    #             piece = board[r][c]
    #             if piece > 1 and piece not in piece_mapping:
    #                 piece_mapping[piece] = 2 if piece == 2 else next_piece_number
    #                 if piece != 2:
    #                     next_piece_number += 1
    #
    #     for r in range(self.height):
    #         for c in range(self.width):
    #             if board[r][c] in piece_mapping:
    #                 board[r][c] = piece_mapping[board[r][c]]
    #
    #     return self

    def normalize(self):
        cloned_puzzle = self.clone()
        board = cloned_puzzle.board

        piece_mapping = {}
        next_piece_number = 3

        for r in range(self.height):
            for c in range(self.width):
                piece = board[r][c]

                if piece > 1 and piece not in piece_mapping:
                    piece_mapping[piece] = 2 if piece == 2 else next_piece_number
                    if piece != 2:
                        next_piece_number += 1

            for c in range(self.width):
                if board[r][c] in piece_mapping:
                    board[r][c] = piece_mapping[board[r][c]]

        return cloned_puzzle

    def random_walk(self, steps):
        new_puzzle = self.clone()

        print("Initial Board State:")
        new_puzzle.display_state()

        for i in range(steps):
            available_moves = new_puzzle.get_available_moves()

            if not available_moves:
                print("No more valid moves. Stopping random walk.")
                break

            move = random.choice(available_moves)
            piece, direction = move
            print(f"Step {i + 1}: Moving piece {piece} {direction}")

            new_puzzle = new_puzzle.apply_move(piece, direction)
            new_puzzle = new_puzzle.normalize()
            new_puzzle.display_state()

            if new_puzzle.is_solved():
                print("Puzzle solved during random walk!")
                break

        return new_puzzle

    def bfs(self):
        start_time = time.time()
        queue = deque([(self, [])])
        visited = set()
        nodes_explored = 0
        while queue:
            state, path = queue.popleft()

            if state.is_solved():
                end_time = time.time()
                for move in path:
                    print(move)
                state.display_state()
                print(f"Number of Nodes Visited: {nodes_explored}")
                print(f"Total Search Time: {(end_time - start_time) * 1000:.2f} ms")
                print(f"Solution Path Length: {len(path)}")
                return path

            state_tuple = tuple(map(tuple, state.board))

            if state_tuple in visited:
                continue
            visited.add(state_tuple)
            nodes_explored += 1

            for move in state.get_available_moves():
                piece, direction = move
                new_state = state.apply_move(piece, direction).normalize()
                queue.append((new_state, path + [move]))

        print("No solution found.")
        return None

    def dfs(self):
        start_time = time.time()

        stack = [(self, [])]
        visited = set()
        nodes_explored = 0

        while stack:
            state, path = stack.pop()

            norm = tuple(map(tuple, state.normalize().board))

            if norm in visited:
                continue
            visited.add(norm)
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

            for move in reversed(state.get_available_moves()):
                piece, direction = move
                new_state = state.apply_move(piece, direction)
                stack.append((new_state, path + [move]))

        print("No solution found.")
        return None

    def depth_limited_search(self, depth_limit):
        open_list = deque([(self, [], 0)])
        closed = {}
        nodes_explored = 0

        while open_list:
            state, path, depth = open_list.pop()

            if state.is_solved():
                return path, state, nodes_explored

            state_tuple = tuple(map(tuple, state.normalize().board))

            if depth <= 50:
                existing_depth = closed.get(state_tuple)
                if existing_depth is not None and existing_depth <= depth:
                    continue
                closed[state_tuple] = depth
            else:
                if state_tuple in closed:
                    continue
                closed[state_tuple] = True

            nodes_explored += 1

            if depth < depth_limit:
                for move in state.get_available_moves():
                    piece, direction = move
                    new_state = state.apply_move(piece, direction)
                    open_list.append((new_state, path + [move], depth + 1))

        return None, None, nodes_explored

    def ids(self):
        start_time = time.time()
        max_depth_limit = 1000

        for depth_limit in range(max_depth_limit + 1):
            print(f"Trying depth limit: {depth_limit}")
            path, final_state, nodes_explored = self.depth_limited_search(depth_limit)

            if path is not None:
                end_time = time.time()
                print(f"Solution found at depth {len(path)}")
                for move in path:
                    print(move)
                final_state.display_state()
                print(f"Number of Nodes Visited: {nodes_explored}")
                print(f"Total Search Time: {(end_time - start_time) * 1000:.2f} ms")
                print(f"Solution Path Length: {len(path)}")
                return path

        print("No solution found within this max_depth_search_limit.")
        return None

    # def heuristic(self):
    #     if self.is_solved():
    #         return 0
    #
    #     goal_positions = [(r, c) for r in range(self.height) for c in range(self.width) if self.board[r][c] == -1]
    #
    #     if not goal_positions:
    #         print("WARNING: No goal (-1) found on the board!")
    #         return float("inf")
    #
    #     goal_r, goal_c = goal_positions[0]
    #     master_positions = [(r, c) for r in range(self.height) for c in range(self.width) if self.board[r][c] == 2]
    #
    #     if not master_positions:
    #         print("WARNING: No master brick (2) found on the board!")
    #         return float("inf")
    #
    #     min_distance = min(abs(r - goal_r) + abs(c - goal_c) for r, c in master_positions)
    #     blocking_pieces = 0
    #     for r, c in master_positions:
    #         if goal_r == r:
    #             blocking_pieces += sum(1 for col in range(min(c, goal_c), max(c, goal_c))
    #                                    if self.board[r][col] not in {0, 2, -1})
    #         if goal_c == c:
    #             blocking_pieces += sum(1 for row in range(min(r, goal_r), max(r, goal_r))
    #                                    if self.board[row][c] not in {0, 2, -1})
    #
    #     lambda_weight = 1.
    #     return min_distance + lambda_weight * blocking_pieces

    def heuristic(self):
        if self.is_solved():
            return 0

        # Locate goal (-1) and master piece (2)
        goal_positions = [(r, c) for r in range(self.height) for c in range(self.width) if self.board[r][c] == -1]
        master_positions = [(r, c) for r in range(self.height) for c in range(self.width) if self.board[r][c] == 2]

        if not goal_positions or not master_positions:
            return float("inf")  # No solution possible

        goal_r, goal_c = goal_positions[0]

        # 1️⃣ Manhattan Distance (Baseline)
        min_distance = min(abs(r - goal_r) + abs(c - goal_c) for r, c in master_positions)

        # 2️⃣ Count Blocking Pieces in the Path
        blocking_pieces = 0
        bottleneck_penalty = 0  # Extra penalty for blocks closer to goal

        for r, c in master_positions:
            if goal_r == r:  # Horizontal path
                for col in range(min(c, goal_c), max(c, goal_c)):
                    if self.board[r][col] not in {0, 2, -1}:
                        blocking_pieces += 1
                        bottleneck_penalty += (abs(col - goal_c) + 1)  # More weight if closer to goal
            if goal_c == c:  # Vertical path
                for row in range(min(r, goal_r), max(r, goal_r)):
                    if self.board[row][c] not in {0, 2, -1}:
                        blocking_pieces += 1
                        bottleneck_penalty += (abs(row - goal_r) + 1)

        # 3️⃣ Movability Penalty (Reduce weight for easy-to-move blockers)
        movability_penalty = 0
        for piece in set(self.board[r][c] for r, c in master_positions):
            if piece not in {0, 2, -1}:
                moves = len(self.get_available_moves())  # More moves = less penalty
                movability_penalty += (1 / (moves + 1))

        # 🔹 New Weight Tuning
        lambda_1 = 0.1  # Base penalty for blocking pieces
        lambda_2 = 0.3 # Heavier penalty for deep bottlenecks
        lambda_3 = 0.6 # Penalize hard-to-move pieces

        return min_distance + (lambda_1 * blocking_pieces) + (lambda_2 * bottleneck_penalty) + (
                    lambda_3 * movability_penalty)

    def astar(self):
        start_time = time.time()
        priority_queue = []
        counter = itertools.count()

        initial_state = self.normalize()
        heapq.heappush(priority_queue, (0, 0, next(counter), initial_state, []))

        visited = set()
        nodes_explored = 0

        while priority_queue:
            _, g, _, state, path = heapq.heappop(priority_queue)

            if state.is_solved():
                end_time = time.time()
                for move in path:
                    print(move)
                state.display_state()
                print(f"Number of Nodes Visited: {nodes_explored}")
                print(f"Total Search Time: {(end_time - start_time) * 1000:.2f} ms")
                print(f"Solution Path Length: {len(path)}")
                return path

            normalized_state = state.normalize()
            state_tuple = tuple(map(tuple, normalized_state.board))

            if state_tuple in visited:
                continue
            visited.add(state_tuple)
            nodes_explored += 1

            for move in state.get_available_moves():
                piece, direction = move
                new_state = state.apply_move(piece, direction)
                g_new = g + 1
                h_new = new_state.heuristic()
                f_new = g_new + h_new
                heapq.heappush(priority_queue, (f_new, g_new, next(counter), new_state, path + [move]))

        print("No solution found.")
        return None

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "done":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        print(puzzle.is_solved())
    elif len(sys.argv) == 3 and sys.argv[1] == "print":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        puzzle.display_state()
    elif len(sys.argv) == 3 and sys.argv[1] == "availableMoves":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        for move in puzzle.get_available_moves():
            print(move)
    elif len(sys.argv) == 4 and sys.argv[1] == "applyMove":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        move_arg = sys.argv[3].replace("(", "").replace(")", "").replace("'", "").replace('"', "")
        move_parts = move_arg.split(",")
        try:
            piece = int(move_parts[0].strip())
            direction = move_parts[1].strip()
            new_puzzle = puzzle.apply_move(piece, direction)
            new_puzzle.display_state()
        except Exception as e:
            print(f"Error: Invalid move format. Expected (piece, 'direction'). Got: {sys.argv[3]}")
            sys.exit(1)
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        puzzle1 = SlidingBrickPuzzle.load_state(sys.argv[2])
        puzzle2 = SlidingBrickPuzzle.load_state(sys.argv[3])
        print(puzzle1.compare_states(puzzle2))
    elif len(sys.argv) == 3 and sys.argv[1] == "norm":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        normalized_puzzle = puzzle.normalize()
        normalized_puzzle.display_state()
    elif len(sys.argv) == 4 and sys.argv[1] == "random":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        steps = int(sys.argv[3])
        new_puzzle = puzzle.random_walk(steps)
    elif len(sys.argv) == 3 and sys.argv[1] == "bfs":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        puzzle.bfs()
    elif len(sys.argv) == 3 and sys.argv[1] == "dfs":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        puzzle.dfs()
    elif len(sys.argv) == 3 and sys.argv[1] == "ids":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        puzzle.ids()
    elif len(sys.argv) == 3 and sys.argv[1] == "astar":
        puzzle = SlidingBrickPuzzle.load_state(sys.argv[2])
        puzzle.astar()
    else:
        print("Usage:")
        print("  python sbp.py print <filename>       # Print board")
        print("  python sbp.py done <filename>        # Check if solved")
        print("  python sbp.py availableMoves <filename>                      # Show available moves")
        print("  python sbp.py applyMove <filename> \"(piece, 'direction')\"    # Apply move")
        print("  python sbp.py compare <filename1> <filename2>                # Compare two puzzles' states")
        print("  python sbp.py norm <filename>        # normalize the board")
        print("  python sbp.py random <filename>      # random walk")
        print("  python sbp.py bfs <filename>         # Run BFS search")
        sys.exit(1)
