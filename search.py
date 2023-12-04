import numpy as np
import queue
from game import BoardState, GameSimulator, Rules, Player, AdversarialSearchPlayer
from heapq import heappush, heappop
import copy

class Problem:

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        The form of initial state is:
        ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

        # self.player_idx = initial_board_state[1] 
        self.player_idx = player_idx
        self.memoization_table = {}
        self.best_child_state = None
        self.memoization_table_decoding = {}
        self.best_move = None


    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.
        TODO: You need to set self.search_alg_fnc here
        """
        self.search_alg_fnc = self.secondary_search  # set the def you made for the search dijkstra_search

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()
        # print("self.sim.generate_valid_actions(p)", self.sim.generate_valid_actions(p))
        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    def dijkstra_search(self):
        start_state = self.initial_state
        goal_states = self.goal_state_set
        print('start_state',start_state)
        print("goal_states", goal_states) # goal_states is a set of multiple goal_state
        
        queue = [(0, start_state, None)]  # (cost, state, action) --> cost of the initial state is 0 and no action taken yet, action should be action taken from the initial state
        visited = set()
        

        while queue:
            cost, current_state, action = heappop(queue)   # queue got pop here
            print("cost, current_state, action")
            print(cost, current_state, action)
            # print("queue", queue)
            if current_state in visited:
                continue

            visited.add(current_state)
            # print("visited", visited)

            if current_state in goal_states:
                # Reconstruct the path from start to goal
                print("current_state in goal_states")
                path = [(current_state, action)]
                print("path", path)
                while action is not None:
                    current_state, action = visited[path[-1][0]], path[-1][2] #current_state, action = visited(path[-1][0], path[-1][2])
                    path.append((current_state, action))
                    print("updated path", path)
                path.reverse()
                resulting_path = path[1:]
                print("resulting_path", resulting_path)
                return resulting_path
                # return list(reversed(path[:-1]))
            
            for next_action in self.get_actions(current_state):
                # print("next action ")
                next_state = self.execute(current_state, next_action)
                new_cost = cost + 1  # Assuming all actions have the same cost (1)

                # Add the next state to the queue with the updated cost
                heappush(queue, (new_cost, next_state, next_action))

            # Check if all states have been explored (no valid solution)
            if len(visited) == len(queue):
                break

        # Handle the case when the goal state is the same as the initial state
        if start_state in goal_states:
            return []

        return None  # No path found
    
    def secondary_search(self):
        # Initialize the priority queue and distances map
        # print("we are in secondary_search")
        start = self.initial_state
        pq = queue.PriorityQueue()
        distances = {start: 0}
        prev_state = {} # prev_state = {start: (None,None)}
        player_idx = start[1]
        # print("player_idx", player_idx)
        pq.put((0, start))

        # Dijkstra's algorithm - until there are no more unprocessed states in the queue
        while not pq.empty():
            (dist, state) = pq.get()

            # If this is a goal state, we are done; backtrack the path.
            if self.is_goal(state):
                path = [(state, None)]
                while state is not start:
                    
                    action = prev_state[state][1] # action should be current action 

                    state = prev_state[state][0] # state should be the prev state
                    path.append((state, action))
                    # print("path", path)
                path.reverse()

                return path

            # Add neighbors
            for action in self.get_actions(state):
                next_state = self.execute(state, action)
                next_dist = dist + 1
                if next_state not in distances or next_dist < distances[next_state]:
                    distances[next_state] = next_dist
                    pq.put((next_dist, next_state))
                    prev_state[next_state] = (state, action)

        # If we reach this then there's no path to a goal state
        return None
    def single_piece_distance_heuristics(self,current_x, current_y, goal_x, goal_y): 
        # we need to calculate the heuristic value for all potentail moves for a single piece in a given board_state

        """
        it should take the current state of the game as input and return a heuristic value that reflect the quality of the game stat
        """
        heuristic = abs(current_x - goal_x) + abs(current_y - goal_y)

        return heuristic*10

    def single_ball_distance_heuristics(self,current_x, current_y, goal_x, goal_y):# we need to calculate the heuristic value for all potentail moves for a single ball in a given board_state
        dx= abs(current_x - goal_x) 
        dy = abs(current_y - goal_y)
        heuristic = max(dx, dy)
        return heuristic*10
    # 
    
    def adversarial_search_method(self, state_tup, val_a, val_b, val_c): # NOTE: adversarial_search_method() here is the prediction function based on miniMax
        """
        TODO: 
        Input:
        state_tup = tuple((encoded_state_tup, self.player_idx)) -> example = (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52), 0)
        val_a, val_b, val_c = (1, 2, 3)  --> I think those vals are depth, alpha, beta

        output: return a list of move as a optimal player (action = (relative_idx, encoded position), value)
        (please note: the result list of moves are a set of tuples (relative_idx, encoded position), where relative_idx refers to the piece to move, encoded position is the move position)
        --> a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.

        note: 
        is_goal state vs  is_termination_state is different. In here, we should use is_termination_state(). 

        OH: 
        given a state, you need to decide what move need to do next for a piece. 
        """

   
        def minMax_utility(self, state_tup, depth):
            curr_state, curr_player_idx = state_tup
            print("curr state in utility: ", curr_state)

            def get_weight(piece_idx):
                if piece_idx in [0, 1, 2, 3, 4]:
                    return 1  # Pieces
                elif piece_idx == 5:
                    return 3  # Ball (weighted more)
                elif piece_idx in [6, 7, 8, 9, 10]:
                    return 1  # Pieces (opponent's side)
                else:
                    return 3  # Ball (opponent's side, weighted more)

            def get_goal_y(piece_idx):

                if piece_idx in [0, 1, 2, 3, 4, 5]:
                    return 0  # Pieces
                elif piece_idx in [6, 7, 8, 9, 10, 11]:
                    return 7

            total_score = 0

            for piece_idx, each_piece in enumerate(curr_state):
                
                # print("piece_idx, each_piece")
                # print(piece_idx, each_piece)
                # add memoization_table here for decoding:
                if each_piece in self.memoization_table_decoding:
                    move_x, move_y = self.memoization_table_decoding[each_piece]
                
                move_x, move_y = self.sim.game_state.decode_single_pos(each_piece)
                self.memoization_table_decoding[each_piece] = (move_x, move_y)
                # print(move_x, move_y, )
                goal_x = move_x
                goal_y = get_goal_y(piece_idx)
                # print(move_x, move_y, goal_y)
                if piece_idx in [5,11]: 
                    distance = self.single_ball_distance_heuristics(move_x, move_y, goal_x, goal_y)
                distance = self.single_piece_distance_heuristics(move_x, move_y, goal_x, goal_y)
                weight = get_weight(piece_idx)
                # print("each_piece eval: ", distance * weight)
                total_score += distance  * weight
            print("total_score", total_score)
            return total_score


        def generate_child_states(self, state_tup, current_player):
            print("state_tup, current_player")
            print(state_tup, current_player)
            next_possible_moves = self.get_actions(state_tup)
            print("next_possible_moves")
            print(next_possible_moves)
            child_states = []
            
            

            for each_move in next_possible_moves:
                new_state = list(state_tup[0])
                relative_idx = each_move[0]
                encoded_position = each_move[1]

                if current_player == 0:
                    new_state[relative_idx] = encoded_position
                else:
                    new_state[relative_idx + 6] = encoded_position

                new_state_up = (tuple(new_state), 1- current_player)  # Switch player: 1 - current_player indicates the next player in the same round
                child_states.append((new_state_up, each_move))

            return child_states

        # def if_maximizing(self, state_tup, alpha, beta, depth, current_player):
        
        def miniMax_algo(self, state_tup, alpha, beta, depth, current_player, curr_round_player):
            if state_tup in self.memoization_table:
                return self.memoization_table[state_tup]
            counter = 0
            curr_state = state_tup[0]
            current_player = state_tup[1]
            # print(curr_round_player, current_player)
            

            if depth == 0 or curr_state[5] >= 49 or curr_state[11] <= 6:
                return minMax_utility(self, state_tup, depth)

            all_child_states = generate_child_states(self, state_tup, current_player) #checked
            # print("all_child_states", all_child_states)
            

            if current_player == curr_round_player:  # Maximizing player
                print("Maximizing player", current_player)
                # print("curr_player_idx, current_player") 
                # print(curr_player_idx, current_player) 
                maxEval = float("-inf")
                best_child_state = None
                best_move_list = set()

                for child_state, move in all_child_states:
                    # current_player = child_state[1] # this is always the next player from the parent player
                    # print(curr_round_player, current_player)
                    # exit()
                    eval = miniMax_algo(self, child_state, alpha, beta, depth - 1, 1-current_player, curr_round_player) #current_player
                    # print("eval", eval)
                    counter += 1

                    if eval >= maxEval:
                        # if eval == maxEval:
                        #     best_child_state = max(child_state, best_child_state, key=move[])
                        #     best_move = max(move, best_move)
                        maxEval = eval
                        best_child_state = child_state
                        if eval == maxEval:
                            best_move_list.add(move)
                        # if eval == maxEval: best_move_list[move[1]] = move[0]
                        best_move = move
                        # print("maxEval, best_child_state, best_move")
                        # print(maxEval, best_child_state[0], best_move)

                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break

                self.best_child_state = best_child_state
                self.memoization_table[state_tup] = maxEval
                print("best_move_list", best_move_list)
                self.best_move = best_move
                if current_player == 0:
                    max_move = float("-inf")
                    for each_move in list(best_move_list):
                        if each_move[1] > max_move:
                            max_move = each_move[1]
                            winning = each_move
                else:
                
                    max_move = float("inf")
                    for each_move in list(best_move_list):
                        if each_move[1] < max_move:
                            max_move = each_move[1]
                            winning = each_move
                self.best_move = winning 
                
                return maxEval

            else:  # Minimizing player
                print("Minimizing player", current_player)
                # print("curr_player_idx, current_player") 
                # print(curr_player_idx, current_player) 
                minEval = float("inf")
                best_child_state = None

                for child_state, move in all_child_states:
                    # current_player = child_state[1]
                    eval = miniMax_algo(self, child_state, alpha, beta, depth - 1, 1-current_player, curr_round_player)

                    if eval < minEval:
                        minEval = eval
                        best_child_state = child_state
                        best_move = move
                        # print("minEval, best_child_state, best_move")
                        # print(minEval, best_child_state[0], best_move)

                    beta = min(beta, eval)
                    if beta <= alpha:
                        break

                self.best_child_state = best_child_state
                self.memoization_table[state_tup] = minEval
                self.best_move = best_move
                
                return minEval


        alpha = float("-inf")
        beta =  float("inf")
        depth = 1
        curr_round_player = state_tup[1]
        current_player = curr_round_player
        print("curr_round_player", curr_round_player) # this should be fixed per round
        best_child_eval = miniMax_algo(self, state_tup, alpha, beta, depth, current_player, curr_round_player)
        winning_child_state = self.best_child_state
        wining_move = self.best_move
        print("wining_move, best_child_eval, current_player")
        print(wining_move, best_child_eval, current_player)
        
        return wining_move, best_child_eval
        

        # player_idx = state_tup[1]
        # for i in range(len(state_tup[0])):
        #     if state_tup[0][i] != winning_child_state[0][i]:
        #         if player_idx == 0:
        #             new_action = (i, winning_child_state[0][i])
        #         else:
        #             new_action = (i - 6, winning_child_state[0][i])
        # print("new_action, best_child_eval")
        # print(new_action, best_child_eval)
        # # exit()
        # return new_action, best_child_eval

    



all_actions = [(3, 53), (2, 47), (0, 26), (5, 34), (0, 16), (3, 33), (2, 39), (3, 39), (0, 6), (2, 19), (0, 2), (5, 55), (4, 40), (2, 25), (5, 48), (0, 24), (4, 46)]
white_moves = [(3, 43), (4, 42), (3, 37), (4, 44), (1, 36), (4, 24), (5, 29), (5, 28)]
print(max(list(white_moves)))
# print(min((0, 6),(4, 40)))
# max = -1
# for each_move in white_moves:
#     if each_move[1] > max:
#         max = each_move[1]
#         winning = each_move

# print(winning)