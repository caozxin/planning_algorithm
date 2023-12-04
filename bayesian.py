import numpy as np
import random

class StateGenerator:

    def __init__(self, nrows=8, ncols=7, npieces=10):
        """
        Initialize a generator for sampling valid states from
        an npieces dimensional state space.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.npieces = npieces
        self.rng = np.random.default_rng()

    def sample_state(self):
        """
        Samples a self.npieces length tuple.

        Output:
            Returns a state. A state is as 2-tuple (positions, dimensions), where
             -  Positions is represented as an encoded positions tuple (entries from 0 to nrows*ncols)
                This means positions is a tuple with self.npieces entries, and each
                entry is an integer on the interval [0, self.nrows*self.ncols]
             -  Dimensions is a 2-tuple (self.nrows, self.ncols)

            For example, if the dimensions of the board are 2 rows, 3 columns, and the number of pieces
            is 4, then a valid return state would be ((0,3,2,4), (2,3)) because it has 4 positions, each
            of which is an integer on [0,2*3].
        """
        ## Returns positions in encoded format.
        ## Without loss of generalization, we assume that positions[1:] are fixes; only
        ## positions[0] will be moved
        positions = self.rng.choice(self.nrows*self.ncols, size=self.npieces, replace=False)
        pos = list(self.decode(p) for p in positions)
        # pos = [(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)]
        pos = [(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)]

        return pos, (self.nrows, self.ncols)

    def decode(self, position):
        r = position // self.ncols
        c = position - self.ncols * r
        return (c, r)

def sample_observation(state): # --> likelihood function --> this is clear. DO NOT TOUCH it!
    """
    TODO
    Given a state, sample an observation from it. Specifically, the positions[1:] locations are
    all known, while positions[0] should have a noisy observation applied.

    Input:
        State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        A tuple (position, distribution) where:
         - Position is a sampled postion which is a 2-tuple (c, r), which represents the sampled observation
         - Distribution is a 2D numpy array representing the observation distribution

    NOTE: the array representing the distribution should have a shape of (nrows, ncols)
    """
    positions, (nrows, ncols) = state # all positions are in (c,r)
    observation_probs = np.zeros((nrows, ncols))
    print(observation_probs.shape)
    # Set the probability of the piece's block to 0.6
    piece_position = positions[0] # this is the first position
    # print("piece_position[0] - C , piece_position[1] - R")
    # print(piece_position[0], piece_position[1])
    # observation_probs[piece_position[1], piece_position[0]] = 0.6
    observation_probs[piece_position[1], piece_position[0]] = 0.6 # distribution should be R * C. 

    # Loop through four neighbors of the piece (up, down, left, right neighbors)
    for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor_position = (piece_position[0] + dc, piece_position[1] + dr)

        # Check if the neighbor is outside of the board
        if (
            neighbor_position[0] < 0
            or neighbor_position[0] >= ncols
            or neighbor_position[1] < 0
            or neighbor_position[1] >= nrows
        ):
            observation_probs[piece_position[1], piece_position[0]] += 0.1
        # Check if the neighbor is occupied
        elif neighbor_position in positions[1:]:
            observation_probs[piece_position[1], piece_position[0]] += 0.1
        else:
            # Set the neighbor's probability to 0.1 --> neighbor_position[1] - R, neighbor_position[0] - C
            observation_probs[neighbor_position[1], neighbor_position[0]] = 0.1

    # Sample a position according to observation probs
    sampled_position = np.unravel_index(
        # np.random.choice(np.flatnonzero(observation_probs == observation_probs.max())),
        np.random.choice(np.flatnonzero(observation_probs.flatten() == observation_probs.max())),
        observation_probs.shape,
        # observation_probs.shape[::-1],
    )

    return sampled_position, observation_probs


def initialize_belief(initial_state, style="uniform"):
    """
    TODO
    Create an initial belief, based on the type of belief we want to start with

    Inputs:
        Initial_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        style: an element of the set {"uniform", "dirac"}

    Returns:
        an initial belief, represented by a 2D numpy array with shape (nrows, ncols)

    NOTE:
        The array representing the distribution should have a shape of (nrows, ncols).
        The occupied spaces (if any) should be zeroed out in the belief.
        We define two types of priors: a uniform prior (equal probability over all
        unoccupied spaces), and a dirac prior (which concentrates all the probability
        onto the actual position on the piece).
    
    """

    positions, (nrows, ncols) = initial_state
    L = len(positions) - 1
    # print("L", L, nrows * ncols)
    if style == "uniform":

        total_unoccupied = nrows * ncols - L
        belief = np.ones((nrows, ncols))
        # belief /= (nrows * ncols - len(pos) + 1)
        belief = belief / total_unoccupied

        for pos in positions[1:]:  
            # print("pos", pos)
            belief[pos[1]][pos[0]] = 0.0
            # belief[pos[0]][pos[1]] = 0.0

    elif style == "dirac":
        # Dirac prior: concentrate all the probability onto the actual position of the piece
        belief = np.zeros((nrows, ncols))
        pos = positions[0]
        belief[pos[1], pos[0]] = 1.0
    
    # Check if the total probability is not zero
    belief /= np.sum(belief)
    if np.sum(belief) != 0:
        belief /= np.sum(belief)

    return belief


def belief_update(prior, observation, reference_state):
    nrows, ncols = prior.shape
    posterior = np.zeros((nrows, ncols))
    positions, (nrows, ncols) = reference_state
    new_state = positions.copy(), (nrows, ncols)
    obs_col, obs_row = observation

    # sampled_position, observation_probs = sample_observation(reference_state) # this should be correct
    for row in range(nrows): # 
        for col in range(ncols):
            
            new_state[0][0] = [col, row]
            # print("new_state[0][0]")
            # print(new_state[0][0])
            sampled_position, observation_probs = sample_observation(new_state)

            likelihood = observation_probs[obs_row, obs_col] 
            posterior[row, col] = prior[row, col] * likelihood
            # print("posterior[row, col]", posterior[row, col])
            # print("likelihood", likelihood)

    total_probability = np.sum(posterior) # this section is correct
    if total_probability != 0:
        for pos in reference_state[0][1:]:
            posterior[pos[1], pos[0]] = 0.0

    posterior /= np.sum(posterior)
    
    return posterior

def sample_transition(state, action):  
    positions, (nrows, ncols) = state
    current_position = positions[0]
    new_position = (current_position[0] + action[0], current_position[1] + action[1]) # current_position[0] + action[0] - C; current_position[1] + action[1] - R; 
    if (
        new_position[0] < 0 # check if the new position is out of board
        or new_position[0] >= ncols
        or new_position[1] < 0
        or new_position[1] >= nrows
        or new_position in positions[1:]  # check if the new position is occupied
    ):
        return None, np.zeros((nrows, ncols))

    transition_probabilities = np.ones((nrows, ncols))
    # transition_probabilities[new_position[1], new_position[0]] = 1.0 #transition_probabilities[ new_position[0], new_position[1]] = 1.0
    return new_position, transition_probabilities

def belief_predict(prior, action, reference_state):
    pos, (rows, cols) = reference_state
    dc, dr = action
    posterior = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            new_col = col + dc
            new_row = row + dr
            if 0 <= new_col < cols and 0 <= new_row < rows and (new_col, new_row) not in pos[1:]:
                posterior[new_row, new_col] += prior[row, col]

    if np.sum(posterior) != 0:
        for neighbor in pos[1:]:
            posterior[neighbor[1], neighbor[0]] = 0
    posterior /= np.sum(posterior)
    return posterior

def belief_predict02(prior, action, reference_state):
    positions, (nrows, ncols) = reference_state
    dc, dr = action
    nrows, ncols = prior.shape

    # Initialize the posterior belief
    posterior = np.zeros((nrows, ncols))
    new_position, transition_probabilities = sample_transition(reference_state, action)

    for row in range(nrows):

        for col in range(ncols):
            
            new_col = col + dc
            new_row = row + dr

            if 0 <= new_col < ncols and 0 <= new_row < nrows:
                  # get transition probabilities using sample_transition
                # Consider the transition probabilities when updating the posterior
                posterior[new_row, new_col] += transition_probabilities[new_row, new_col] * prior[row, col]

    total_probability = np.sum(posterior)
    if total_probability != 0:
        for pos in reference_state[0][1:]:
            posterior[pos[1], pos[0]] = 0.0
    posterior /= np.sum(posterior)
    
    return posterior


if __name__ == "__main__":
    gen = StateGenerator()
    initial_state = gen.sample_state() #"initial_state,observation_list,prior_style"
    updated_initial_state = (([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),[(3,4)], "uniform")
    print("initial_state", initial_state)
    obs, dist = sample_observation(initial_state)
    print("initial_state", initial_state)
    print("sample_observation position",obs) # (col, row)
    print("sample_observation distribution", dist)
    b = initialize_belief(initial_state, style="uniform")
    print("initialize_belief")
    print(b)
    b = belief_update(b, obs, initial_state)
    print("belief_update")
    print(b)
    b = belief_predict(b, (1,0), initial_state)
    print("belief_predict")
    print(b)
