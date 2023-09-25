import numpy as np
import queue
from game import BoardState, GameSimulator, Rules
from heapq import heappush, heappop

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

    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx),where encoded_state is a  tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## goal is to set the goal of board state; 
    """ Here is an example:
    
    def my_snazzy_search_algorithm(self):
        ## Some kind of search algorithm
        ## ...
        return solution ## Solution is an ordered list of (s,a)
        
    """
    """
    Output: 
    1) the search method returns a ordered list of (state, action) pairs
    2) 1st state and action == the initial state and action taken from the initial state
    3) last (s,a) pair has s as a goal state, and a=None
    4) other (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.

    Formats:
    1) state is a tuple: (encoded_state, player_idx),where encoded_state is a tuple of 12 integers (mirroring the contents of BoardState.state), and player_idx is 0 or 1
    2) The format of action is a tuple: (relative_idx, position), where relative_idx is a integer between [0, 5] inclusively and position is the encoded position (a integer) where the piece should be moved

    Stop Condition:
        to reach self.is_goal() --> to reach a goal state


    def updated_my_search_algorithm(self):
        ## Some kind of search algorithm
        ## ...
        return solution ## Solution is an ordered list of (s,a)

        if the resulted_state in self.is_goal(state):
            print("the search is over")
            return a ordered list of (state, action) pairs

        --> you need to use GameSimulator.generate_valid_actions(self, player_idx: int)
        --> you also need to calculate the distance between the ball and the node and track it in the
        priority queue()
        --> our goal is to find the squence of state-action pairs with shortest steps (distance/length)
        from initial state to goal state. 


    """
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

            # Discard if distance is longer; this can happen when we found a shorter path
            # to this state after adding it to the queue.
            # if distances[state] < dist:
            #     continue

            # If this is a goal state, we are done; backtrack the path.
            if self.is_goal(state):
                path = [(state, None)]
                while state is not start:
                    
                    # print("start", start)
                    # print("state", state)
                    # action = prev_state[state][1] 
                    # state = prev_state[state][0]
                    # path.append((state, action))
                    action = prev_state[state][1] # action should be current action 
                    # print("*****  player_idx", state[1])
                    # print("action", action)
                    
                    state = prev_state[state][0] # state should be the prev state
                    path.append((state, action))
                    # print("path", path)


                path.reverse()

                # print("result", path )
                # return path[-1:] 
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

# Create a BoardState object
# board = BoardState()
initial_state = BoardState()

goal_state = BoardState() #
goal_state.update(0,14)
goal_state.update(9,38)

# print(initial_state.state, goal_state.state) # --> (0,14)
player_idx = 0

# start_state = ((1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52), 0)
# goal_state = ((14, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52), 0)

# problem = GameStateProblem(initial_state, goal_state, player_idx)
# solution = problem.set_search_alg
# problem.secondary_search()

#output_path = [(start_state, (0,14)), (goal_state, None)]