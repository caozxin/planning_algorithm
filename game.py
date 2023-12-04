import numpy as np
from collections import deque
# from search import Problem, GameStateProblem

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        
        self.N_COLS = 7
        self.N_ROWS = 8
                
        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52]) #([1,2,3,4,5,3,50,51,52,53,54,52], False), ## Initial State
        self.decode_state = [self.decode_single_pos(d) for d in self.state]


    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """

        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])


    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive --> for example, (0,0) == 0 and  (6, 7) == 55.

        TODO: You need to implement this.
        """
        # Ensure the input is within valid bounds (0 <= col <= 7, 0 <= row <= 7)
        col, row = cr
        # if not (0 <= col <= 7) or not (0 <= row <= 7):
        #     raise ValueError("Input coordinates out of bounds")

        # Encode the coordinate into a single integer in the interval [0, 55]
        # encoded_value = col + row * 8
        encoded_value = row * 7 + col
        # print("encoded_value", encoded_value)
        if 0 <= encoded_value <= 55:
            return encoded_value
        else:
            raise ValueError("Input coordinates are out of range")

        # raise NotImplementedError("TODO: Implement this function")

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer between  interval [0, 55] inclusive into a coordinate on the board which has 8 rows and 7 columns: Z -> (col, row), 

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)

        TODO: You need to implement this.
        """
        
        if 0 <= n and n <= 55:
        # Calculate the column and row based on the given integer n
            col = n % 7
            row = n // 7

        # Return the coordinates as a tuple (col, row)
            # print('coordinates', (col, row))
            return (col, row)
        else:
            # return False
            # self.is_valid()
            # print("after update", self.is_valid())
            return False
            raise ValueError("Input integer must be in the interval [0, 57] inclusive")
        # raise NotImplementedError("TODO: Implement this function")

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board which has 8 rows and 7 columns.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        
        TODO: You need to implement this.
        """

        # Iterate through the decoded state to check for terminal conditions
        # print("self.decode_state", self.decode_state)
        # print("calling is_termination_state()")
        if self.is_valid() == True: 
            # print("is_valid == True")
            i = 0 
            for (col, row ) in self.decode_state:
                
                # print((col, row), i)
                if i == 5 and row == 7: # if white ball reachs top row
                    # print("is_termination_state == True")
                    return True
                elif i == 11 and row == 0: # if black ball reachs botton row
                    ("is_termination_state == True")
                    return True
                i += 1

        # If no player's ball has reached the opposite side, it's not a termination state
        
        return False

        raise NotImplementedError("TODO: Implement this function")

    def is_valid(self):
        """
        Checks if a board state is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constraints that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)
        """
        # Check if all elements of self.state are within the valid range [0, 55]
        if not all(0 <= value <= 55 for value in self.state):
            # print("out of range")
            return False

        # Check for out of bounds
        for value in self.state:
            col, row = self.decode_single_pos(value)
            if not (0 <= col < 7 and 0 <= row < 8):
                # print("out of bound")
                return False

        
        # print(self.state[:5] ,self.state[6:11])
        white_blocks = self.state[:5]
        black_blocks = self.state[6:11]
        white_ball = self.state[5]
        black_ball = self.state[11]
        blocks_zone = np.concatenate((self.state[:5],self.state[6:11]), axis = 0 )

        # Check for overlap
        # print("blocks_zone", blocks_zone)
        if len(set(blocks_zone)) != len(blocks_zone):
            # print("set(self.state)", set(self.state))
            # print("overlap")
            return False
        # check if white/black ball is out of its block_zone:
        if white_ball not in white_blocks or black_ball not in black_blocks:
            # print("ball out of blocks zone")
            return False

        # print("is_vali self == True")
        return True

        # raise NotImplementedError("TODO: Implement this function")

class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        
        TODO: You need to implement this.
        """
        # print('board_state, piece_idx')
        # print(board_state, piece_idx)
        white_blocks = board_state.state[:5]
        black_blocks = board_state.state[6:11]
        white_ball = board_state.state[5]
        black_ball = board_state.state[11]
        
        # Get the current position of the block piece
        current_position = board_state.state[piece_idx]
        # print("current_position", current_position)
        # # only move block pieces:
        # print("board_state.white_blocks", white_blocks)
        # print("board_state.black_blocks", black_blocks)

        #two preconditions: 
        if current_position not in white_blocks and current_position not in black_blocks:
            # print(" only move blocks!")
            return []
        
        if current_position == white_ball or current_position == black_ball:
            # print("block should not hold the ball!")
            return []
        

        # Decode the current position to obtain its (col, row) coordinates
        current_col, current_row = board_state.decode_single_pos(current_position)

        # Define possible knight move offsets (L-shape)
        knight_move_offsets = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

        # Calculate all possible new positions for the block piece
        possible_new_positions = []
        for offset in knight_move_offsets:
            new_col = current_col + offset[0]
            new_row = current_row + offset[1]

            # Check if the new position is within the bounds of the board and unoccupied
            if 0 <= new_col < board_state.N_COLS and 0 <= new_row < board_state.N_ROWS:
                new_position = board_state.encode_single_pos((new_col, new_row))

                # Check if the new position is unoccupied (not in the state)
                if new_position not in board_state.state:
                    possible_new_positions.append(new_position)
        # print("possible_new_positions", possible_new_positions)
        return possible_new_positions


    @staticmethod
    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for player_idx in the board_state.

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        """
        # Initialize the set of valid move positions for the ball
        valid_move_positions = set()
        white_blocks = board_state.state[:5]
        black_blocks = board_state.state[6:11]
        white_ball = board_state.state[5]
        black_ball = board_state.state[11]

        # Get the current position of the player's ball
        current_ball_position = white_ball if player_idx == 0 else black_ball
        current_blocks_zone = white_blocks if player_idx == 0 else black_blocks
        current_obstacles_zone = black_blocks if player_idx == 0 else white_blocks

        # Define possible movement directions for the ball (similar to queen in chess)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        # Create a queue for BFS traversal
        # queue = deque(current_ball_position)
        queue = deque([current_ball_position])
        # print("queue", queue)
        

        # Set to keep track of visited positions
    
        visited = set()

        # Perform BFS to find valid move positions
        while len(queue) > 0:
            position = queue.popleft()
            # print("queue", queue)
            visited.add(position)
            
            # print("position: ", position )
            
            # Check if the current position is within the bounds of the board
            if 0 <= position < 56:
                
                # Add the position to the set of valid move positions
                # visited.add(position)

                # Explore all possible directions of movement
                
                current_col, current_row = board_state.decode_single_pos(position)
                
                one_direction = set()
                for dx, dy in directions:
                    
                    # print("one_direction", one_direction)
                    new_x = current_col + dx   #(position % 7) + dx
                    new_y = current_row + dy #(position // 7) + dy
                    # new_position = new_x + (new_y * 7)
                    # print("new_position", new_position)
                    
                    # Continue moving in the current direction until an obstruction or edge is reached
                    while 0 <= new_x < board_state.N_COLS and 0 <= new_y < board_state.N_ROWS: # this is the 2nd for loop
                        
                        new_position = new_x + (new_y * 7)
                        # print("new_position", new_position)
                        one_direction.add(new_position)
                        

                        # Check if the new position is already visited
                        if new_position in visited: 
                            break

                        
                        # Check if the new position is blocked by a block piece of any color
                        if new_position in current_obstacles_zone: #board_state.state[:11]:
                            # print("current_obstacles_zone")
                            break
                        
                        
                        if new_position == current_ball_position: 
                            # print("the same as current_ball_position")
                            break
 
                        if new_position in current_blocks_zone:
                            # print(" in current_blocks_zone")
                            # print(new_position, current_blocks_zone)
                            queue.append((new_position))  # you can check if new_position already in queue
                            valid_move_positions.add(new_position)
                            # print("valid_move_positions", valid_move_positions)

                    #     # Update coordinates for the next step in the same direction
                        
                        new_x += dx
                        new_y += dy
                        # print("new_x, new_y," , new_x, new_y)
                        
                        # visited.add(new_position)
                    

                        # print("queue",queue)
                        # print("visited", visited)
                        # visited += each_direction # keep track of those visited
                        
                # exit()
                
        # Convert the set of valid move positions to a list and return it

        # valid_move_positions_list = list(valid_move_positions)
        # print(" ********************         valid_move_positions_list", valid_move_positions_list)
        return valid_move_positions


class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds # self.current_round = -1
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            # if self.current_round >= 5: # NOTE remove this when you fix the heuristic value function! 
            #     break
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")
            print("validate_action", action )
            print("current player_idx", player_idx)
            if not self.validate_action(action, player_idx):
                
                print("validate_action", action )
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
            
        TODO: You need to implement this.
        """
        valid_actions = set()
        policy = Rules()
        white_blocks = self.game_state.state[:5]
        black_blocks = self.game_state.state[6:11]
        white_ball = self.game_state.state[5]
        black_ball = self.game_state.state[11]

        # Get the player's pieces based on player_idx  -> game_state == board_state
        if player_idx == 0:
            block_pieces = white_blocks
            ball_piece = white_ball
            piece_def_idx = 0
        elif player_idx == 1: 
            block_pieces = black_blocks
            ball_piece = black_ball
            piece_def_idx = 6
        
        

        # Loop through the player's block pieces and generate valid actions
        for relative_idx, piece_encoded_position in enumerate(block_pieces):
            # print("block_pieces", block_pieces, ball_piece)
            # print("relative_idx, piece_encoded_position", relative_idx, piece_encoded_position)
            
            # Generate possible moves for the piece
            # if piece_encoded_position != ball_piece: # not holding the ball
            piece_idx = relative_idx + piece_def_idx
            possible_moves = policy.single_piece_actions(self.game_state, piece_idx)
            # possible_moves = self.game_state.generate_possible_moves(piece_encoded_position)
            # print("possible_moves", possible_moves)
            
                # Add each possible move as a valid action
            for move in possible_moves:
                valid_actions.add((relative_idx, move))
            # print("valid_actions", valid_actions)

        #adding possible ball moves:
        ball_moves =  policy.single_ball_actions(self.game_state, player_idx)
        # print("ball_moves", ball_moves)
        
        if len(ball_moves) > 0:
            for each_move in ball_moves:
                valid_actions.add((5, each_move))
        # print("valid_actions", valid_actions)
        # exit()

        return valid_actions
        raise NotImplementedError("TODO: Implement this function")

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        
        TODO: You need to implement this.
        """
        print("action", action)
        print("generate_valid_actions", self.generate_valid_actions(player_idx))
        if action in self.generate_valid_actions(player_idx):
            return True
        else:
            raise ValueError("For each case that an action is not valid, specify the reason that the action is not valid in this ValueError.")

        

    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)


class Player:
    def __init__(self, policy_fnc):
        self.policy_fnc = policy_fnc
    def policy(self, decode_state):
        pass

class AdversarialSearchPlayer(Player):
    def __init__(self, gsp, player_idx):
        """
        You can customize the signature of the constructor above to suit your needs.
        In this example, in the above parameters, gsp is a GameStateProblem, and
        gsp.adversarial_search_method is a method of that class.

        test command: python test_search.py -k test_adversarial_search

        Note:
        1) we will only grade the player class called AdversarialSearchPlayer
        located in the game.py file. 
        2) Your adversarial algorithms should be added to search.py under GameStateProblem; 
        3) your players will then make a call to the appropriate algorithm

       
        Both PlayerAlgorithmA  and PlayerAlgorithmB will be passed on as AdversarialSearchPlayer(Player) in two parameters: (GameStateProblem(b1, b1, 0) and player_idx = (0), 
        where GameStateProblem(b1, b1, 0) = GameStateProblem(initial_board_state, goal_board_state, player_idx)
        Here, PlayerA is the maximing player (PlayerA is the current turn player),
        and PlayerB is the minixing player (PlayerB is the next turn player)

        """
        
        super().__init__(gsp.adversarial_search_method)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx

    def policy(self, decode_state):
        """
        Here, the policy of the player is to consider the current decoded game state
        and then correctly encode it and provide any additional required parameters to the
        assigned policy_fnc (which in this case is gsp.adversarial_search_method), and then
        return the result of self.policy_fnc
        """
        encoded_state_tup = tuple( self.b.encode_single_pos(s) for s in decode_state )
        state_tup = tuple((encoded_state_tup, self.player_idx))
        val_a, val_b, val_c = (1, 2, 3)
        return self.policy_fnc(state_tup, val_a, val_b, val_c)
    
    # def PlayerAlgorithmA(self, player_idx):
        
        
    #     # return AdversarialSearchPlayer(self, player_idx=0)
    #     # return self.adversarial_search_method()
    #     return self
    
    # def PlayerAlgorithmB(self, player_idx):

    #     """
    #     Similar to PlayerAlgorithmA,  PlayerAlgorithmB should allow to pass in two parameters: (GameStateProblem(b1, b1, 0) and player_idx = (1), 
    #     however, both share the same GameStateProblem(b1, b1, 0) = GameStateProblem(initial_board_state, goal_board_state, player_idx)
    #     And PlayerB is the next turn player, thus, PlayerB is the miniing player here. 

    #     """
    #     return


# # Create a BoardState object
# board = BoardState()
# # print("board.state", board.state)
# curr_rules = Rules()
# # curr_rules.single_piece_actions(board, 10)
# # print("single_piece_actions", curr_rules.single_piece_actions(board, 10))
# # curr_rules.single_ball_actions(board, 0)
# new_game = GameSimulator(2)
# print(new_game.generate_valid_actions(0))