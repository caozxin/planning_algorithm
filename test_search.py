import numpy as np
import queue
import pytest
from game import BoardState, GameSimulator, Rules, Player, AdversarialSearchPlayer
from search import GameStateProblem
import os, pathlib


# os.chdir( pathlib.Path.cwd() / 'test_search.py' )

pytest.main() # command to run pytest: python test_search.py

class TestSearch:

    def test_game_state_goal_state(self):
        b1 = BoardState()
        gsp = GameStateProblem(b1, b1, 0)

        sln = gsp.search_alg_fnc()
        ref = [(tuple((tuple(b1.state), 0)), None)]

        assert sln == ref

    ## NOTE: If you'd like to test multiple variants of your algorithms, enter their keys below
    ## in the parametrize function. Your set_search_alg should then set the correct method to
    ## use.
    # test command: python test_search.py -k test_adversarial_search -s
    
    #(14, 21, 22, 28, 29, 22, 11, 20, 34, 48, 55, 55),"BLACK" | (44, 37, 46, 41, 40, 41, 1, 2, 52, 4, 5, 52),"WHITE"
    @pytest.mark.parametrize("p1_class,p2_class,encoded_state_tuple,exp_winner,exp_stat", 
                             [(AdversarialSearchPlayer, AdversarialSearchPlayer,(14, 21, 22, 28, 29, 22, 7, 20, 34, 48, 55, 55),"BLACK", "No issues")])
    def test_adversarial_search(self, p1_class, p2_class, encoded_state_tuple, exp_winner,
    exp_stat):
        
        b1 = BoardState()
        b1.state = np.array(encoded_state_tuple)
        b1.decode_state = b1.make_state()
        print('Welcome! GameStateProblem(b1, b1, 0)', GameStateProblem(b1, b1, 0))

        players = [
        p1_class(GameStateProblem(b1, b1, 0), 0), # PlayerAlgorithmA should allow to pass in two parameters: (GameStateProblem(b1, b1, 0) and player_idx = (0)
        p2_class(GameStateProblem(b1, b1, 0), 1)]

        sim = GameSimulator(players)
        sim.game_state = b1
        rounds, winner, status = sim.run()
        assert winner == exp_winner and status == exp_stat

    @pytest.mark.parametrize("p1_class,p2_class,encoded_state_tuple,exp_winner,exp_stat", [
    ## 1-step wins - AlphaBeta, AlphaBeta
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 40, 55, 40, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 39, 55, 39, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 38, 55, 38, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 36, 55, 36, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 35, 55, 35, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 34, 55, 34, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 48, 55, 48, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 47, 55, 47, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 45, 55, 45, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 44, 55, 44, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 43, 55, 43, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 42, 55, 42, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 29, 55, 29, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 21, 55, 21, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 30, 55, 30, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 31, 55, 31, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 25, 55, 25, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 27, 55, 27, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 20, 55, 20, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 19, 55, 19, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 13, 55, 13, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  6, 55,  6, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 14, 55, 14, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  7, 55,  7, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  0, 55,  0, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    ## 1-step wins
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 40, 55, 40, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 39, 55, 39, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 38, 55, 38, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 36, 55, 36, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 35, 55, 35, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 34, 55, 34, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 48, 55, 48, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 47, 55, 47, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 45, 55, 45, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 44, 55, 44, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 43, 55, 43, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 42, 55, 42, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 29, 55, 29, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 21, 55, 21, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 30, 55, 30, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 31, 55, 31, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 25, 55, 25, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 27, 55, 27, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 20, 55, 20, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 19, 55, 19, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 13, 55, 13, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  6, 55,  6, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 14, 55, 14, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  7, 55,  7, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  0, 55,  0, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    ## 1-step wins
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 40, 55, 40, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 39, 55, 39, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 38, 55, 38, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 36, 55, 36, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 35, 55, 35, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 34, 55, 34, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 48, 55, 48, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 47, 55, 47, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 45, 55, 45, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 44, 55, 44, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 43, 55, 43, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 42, 55, 42, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 29, 55, 29, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 21, 55, 21, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 30, 55, 30, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 31, 55, 31, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 25, 55, 25, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 27, 55, 27, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 20, 55, 20, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 19, 55, 19, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 13, 55, 13, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  6, 55,  6, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46, 14, 55, 14, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  7, 55,  7, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (49, 37, 46,  0, 55,  0, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 11, 20, 34, 48, 55, 55), "BLACK", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 7, 20, 34, 48, 55, 55), "BLACK", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 15, 20, 34, 48, 55, 55), "BLACK", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 17, 20, 34, 48, 55, 55), "BLACK", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 9, 20, 34, 48, 55, 55), "BLACK", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 19, 20, 34, 48, 55, 55), "BLACK", "No issues"),
    (AdversarialSearchPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 13, 20, 34, 48, 55, 55), "BLACK", "No issues"),

])  


    def test_algorithm_one(self, p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat):
        b1 = BoardState()
        b1.state = np.array(encoded_state_tuple)
        b1.decode_state = b1.make_state()
        players = [
            p1_class(GameStateProblem(b1, b1, 0), 0),
            p2_class(GameStateProblem(b1, b1, 0), 1),
        ]
        sim = GameSimulator(players)
        sim.game_state = b1
        rounds, winner, status = sim.run()
        assert winner == exp_winner and status == exp_stat
    
    
    
    @pytest.mark.parametrize("alg", ["", ""])
    def test_game_state_problem(self, alg):
        """
        Tests search based planning
        """
        b1 = BoardState()
        b2 = BoardState()
        b2.update(0, 14)

        gsp = GameStateProblem(b1, b2, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()

        ## Single Step
        ref = [(tuple((tuple(b1.state), 0)), (0, 14)), (tuple((tuple(b2.state), 1)), None)]
        assert sln == ref

        b2 = BoardState()
        b2.update(0, 23)
        
        gsp = GameStateProblem(b1, b2, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()

        ## Two Step:
        ## (0, 14) or (0, 10) -> (any) -> (0, 23) -> (undo any) -> (None, goal state)

        #print(gsp.goal_state_set)
        #print(sln)
        assert len(sln) == 5 ## Player 1 needs to move once, then move the piece back
        assert sln[0] == (tuple((tuple(b1.state), 0)), (0, 14)) or sln[0] == (tuple((tuple(b1.state), 0)), (0, 10))
        assert sln[1][0][1] == 1
        assert sln[2][1] == (0, 23)
        assert sln[4] == (tuple((tuple(b2.state), 0)), None)

    def test_initial_state(self):
        """
        Confirms the initial state of the game board
        """
        board = BoardState()
        assert board.decode_state == board.make_state()

        ref_state = [(1,0),(2,0),(3,0),(4,0),(5,0),(3,0),(1,7),(2,7),(3,7),(4,7),(5,7),(3,7)]

        assert board.decode_state == ref_state

    def test_generate_actions(self):
        sim = GameSimulator(None)
        generated_actions = sim.generate_valid_actions(0)
        assert (0,6) not in generated_actions
        assert (4,0) not in generated_actions

    ## NOTE: You are highly encouraged to add failing test cases here
    ## in order to test your validate_action implementation. To add an
    ## invalid action, fill in the action tuple, the player_idx, the
    ## validity boolean (would be False for invalid actions), and a
    ## unique portion of the descriptive error message that your raised
    ## ValueError should return. For example, if you raised:
    ## ValueError("Cannot divide by zero"), then you would pass some substring
    ## of that description for val_msg.
    @pytest.mark.parametrize("action,player,is_valid,val_msg", [
        ((0,14), 0, True, ""),
        ((0,16), 0, True, ""),
        ((0,10), 0, True, ""),
        ((5,1), 0, True, ""),
        ((5,2), 0, True, ""),
        ((5,4), 0, True, ""),
        ((5,5), 0, True, ""),
    ])
    def test_validate_action(self, action, player, is_valid, val_msg):
        sim = GameSimulator(None)
        if is_valid:
            assert sim.validate_action(action, player) == is_valid
        else:
            with pytest.raises(ValueError) as exinfo:
                result = sim.validate_action(action, player)
            assert val_msg in str(exinfo.value)
        

    @pytest.mark.parametrize("state,is_term", [
        ([1,2,3,4,5,3,50,51,52,53,54,52], False), ## Initial State
        ([1,2,3,4,5,55,50,51,52,53,54,0], False), ## Invalid State
        ([1,2,3,4,49,49,50,51,52,53,54,0], False), ## Invalid State
        ([1,2,3,4,49,49,50,51,52,53,54,54], True), ## Player 1 wins
        ([1,2,3,4,5,5,50,51,52,53,6,6], True), ## Player 2 wins
        ([1,2,3,4,5,5,50,4,52,53,6,6], False), ## Invalid State
    ])
    def test_termination_state(self, state, is_term):
        board = BoardState()
        board.state = np.array(state)
        board.decode_state = board.make_state()

        assert board.is_termination_state() == is_term

    def test_encoded_decode(self):
        board = BoardState()
        assert board.decode_state  == [board.decode_single_pos(x) for x in board.state]

        enc = np.array([board.encode_single_pos(x) for x in board.decode_state])
        assert np.all(enc == board.state)

    def test_is_valid(self):
        board = BoardState()
        assert board.is_valid()

        ## Out of bounds test
        board.update(0,-1)
        assert not board.is_valid()
        
        board.update(0,0)
        assert board.is_valid()
        
        ## Out of bounds test
        board.update(0,-1)
        board.update(6,56)
        assert not board.is_valid()
        
        ## Overlap test
        board.update(0,0)
        board.update(6,0)
        assert not board.is_valid()

        ## Ball is on index 0
        board.update(5,1)
        board.update(0,1)
        board.update(6,50)
        assert board.is_valid()

        ## Player is not holding the ball
        board.update(5,0)
        assert not board.is_valid()
        
        board.update(5,10)
        assert not board.is_valid()

    @pytest.mark.parametrize("state,reachable,player", [
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(0,1),(2,1),(1,2),(1,0)]),
            0
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(2,2)]),
            1
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(0,0)
            ],
            set(),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,3)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(1,2)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
    ]) 
    def test_ball_reachability(self, state, reachable, player):
        board = BoardState()
        board.state = np.array(list(board.encode_single_pos(cr) for cr in state))
        board.decode_state = board.make_state()
        print("current_board_state", board.state)
        predicted_reachable_encoded = Rules.single_ball_actions(board, player)
        print("***********************predicted_reachable_encoded", predicted_reachable_encoded)
        encoded_reachable = set(board.encode_single_pos(cr) for cr in reachable)
        assert predicted_reachable_encoded == encoded_reachable
