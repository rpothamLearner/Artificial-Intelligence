import numpy as np


def part_1_a():
    """Provide probabilities for the letter HMMs outlined below.

    Letters Y and Z.

    See README.md for example probabilities for the letter A.
    See README.md for expected HMMs probabilities.
    See README.md for tuple of states.

    Returns:
        ( prior probabilities for all states for letter Y,
          transition probabilities between states for letter Y,
          emission probabilities for all states for letter Y,
          prior probabilities for all states for letter Z,
          transition probabilities between states for letter Z,
          emission probabilities for all states for letter Z )

        Sample Format (not complete):

        ( {'Y1': prob_of_starting_in_Y1, ...},
          {'Y1': {'Y1': prob_of_transition_from_Y1_to_Y1,
                  'Y2': prob_of_transition_from_Y1_to_Y2}, ...},
          {'Y1': [prob_of_observing_0, prob_of_observing_1], ...},
          {'Z1': prob_of_starting_in_Z1, ...},
          {'Z1': {'Z1': prob_of_transition_from_Z1_to_Z1,
                  'Z2': prob_of_transition_from_Z1_to_Z2}, ...},
          {'Z1': [prob_of_observing_0, prob_of_observing_1], ...} )
    """

    # TODO: complete this function.
#    raise NotImplemented()

    """Letter Y"""
    # prior probabilities for all states for letter Y
    y_prior_probs = { 'Y1': 1.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 }

    # transition probabilities between states for letter Y
    y_transition_probs = {
    'Y1':   { 'Y1': 0.667, 'Y2': 0.333,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 },
    'Y2':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':1.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 },
    'Y3':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 1.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 },
    'Y4':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 1.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 },
    'Y5':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.667,'Y6':0.333, 'Y7':0.0, 'Yend':0.0 },
    'Y6':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':1.0, 'Yend':0.0 },
    'Y7':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.667, 'Yend':0.333 },
    'Yend': { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':1.0 }
    }

    # emission probabilities for all states for letter Y
    y_emission_probs = {
    'Y1'   : [0,1],
    'Y2'   : [1,0],
    'Y3'   : [0,1],
    'Y4'   : [1,0],
    'Y5'   : [0,1],
    'Y6'   : [1,0],
    'Y7'   : [0,1],
    'Yend'  : [0,0]
    }

    """Letter Z"""
    # prior probabilities for all states for letter Z
    z_prior_probs = { 'Z1': 1.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 }

    # transition probabilities between states for letter Z
    z_transition_probs = {
    'Z1':   { 'Z1': 0.667, 'Z2': 0.333,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 },
    'Z2':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':1.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 },
    'Z3':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.667, 'Z4': 0.333, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 },
    'Z4':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 1.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 },
    'Z5':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':1.0, 'Z7':0.0, 'Zend':0.0 },
    'Z6':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':1.0, 'Zend':0.0 },
    'Z7':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':1.0 },
    'Zend': { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':1.0 }
    }

    # emission probabilities for all states for letter Z
    z_emission_probs = {
    'Z1'   : [0,1],
    'Z2'   : [1,0],
    'Z3'   : [0,1],
    'Z4'   : [1,0],
    'Z5'   : [0,1],
    'Z6'   : [1,0],
    'Z7'   : [0,1],
    'Zend'  : [0,0]
    }

    return (y_prior_probs, y_transition_probs, y_emission_probs,
            z_prior_probs, z_transition_probs, z_emission_probs)


def viterbi(evidence_vector, states, prior_probs, transition_probs,
            emission_probs):
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list(int)): List of 0s (Silence) or 1s (Dot/Dash).
            example: [1, 0, 1, 1, 1]
        states (list(string)): List of all states.
            example: ['A1', 'A2', 'A3', 'Aend']
        prior_probs (dict): prior distribution for each state.
            example: {'A1'  : 1.0,
                      'A2'  : 0.0,
                      'A3'  : 0.0,
                      'Aend': 0.0}
        transition_probs (dict): dictionary representing transitions from
            each state to every other state, including self.
            example: {'A1'  : {'A1'  : 0.0,
                               'A2'  : 1.0,
                               'A3'  : 0.0,
                               'Aend': 0.0},
                      'A2'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 1.0,
                               'Aend': 0.0},
                      'A3'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.667,
                               'Aend': 0.333},
                      'Aend': {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.0,
                               'Aend': 1.0}}
        emission_probs (dict): dictionary of probabilities of outputs from
            each state.
            example: {'A1'  : [0.0, 1.0],
                      'A2'  : [1.0, 0.0],
                      'A3'  : [0.0, 1.0],
                      'Aend': [0.0, 0.0]}

    Returns:
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

        Example:
            ( ['A1', 'A2', 'A3', 'A3', 'A3'],
              1.0 )
    """

    

    sequence = []
    probability = 0.0
    
    if evidence_vector == []:
        return [],0.0
    
    if len(evidence_vector) == 1:
        return [],0.0
    
#    print(states)
    V = [{}]
    for state in states:
        V[0][state] = {"p": prior_probs[state]*emission_probs[state][evidence_vector[0]], "prev" : None}
        
    for t in range(1,len(evidence_vector)):
        V.append({})
        for state in states[:-1]:
#            for prev_state in states[:-1]:
#                print("cccccccc")
#                
#                print(prev_state)
#                print(V)
#                print(V[t-1][prev_state]["p"])
#                print("------------------------")
#                print(state)
#                print(transition_probs[prev_state][state])
#                
            max_prev_prob = max(V[t-1][prev_state]["p"]*transition_probs[prev_state][state] for prev_state in states[:-1])
            for prev_state in states[:-1]:
                if V[t-1][prev_state]["p"]*transition_probs[prev_state][state] == max_prev_prob:
                    max_prob = max_prev_prob*emission_probs[state][evidence_vector[t]]
#                    print(state)
                    V[t][state] = {"p":max_prob, "prev":prev_state}
                    print(V[t][state])
                    break
    
    max_prob = max(prob["p"] for prob in V[-1].values())
    prev = None
    
    for state, value in V[-1].items():
        if value["p"] == max_prob:
            sequence.append(state)
            prev = state
            break
        
    for t in range(len(V)-2, -1, -1):
        sequence.insert(0, V[t+1][prev]["prev"])
        prev = V[t+1][prev]["prev"]

    return sequence,max_prob
def part_2_a():
    """Provide probabilities for the NOISY letter HMMs outlined below.

    Letters A, Y, Z, letter pause, word space

    See README.md for example probabilities for the letter A.
    See README.md for expected HMMs probabilities.

    Returns:
        ( list of all states for letter A,
          prior probabilities for all states for letter A,
          transition probabilities between states for letter A,
          emission probabilities for all states for letter A,
          list of all states for letter Y,
          prior probabilities for all states for letter Y,
          transition probabilities between states for letter Y,
          emission probabilities for all states for letter Y,
          list of all states for letter Z,
          prior probabilities for all states for letter Z,
          transition probabilities between states for letter Z,
          emission probabilities for all states for letter Z,
          list of all states for letter pause,
          prior probabilities for all states for letter pause,
          transition probabilities between states for letter pause,
          emission probabilities for all states for letter pause,
          list of all states for word space,
          prior probabilities for all states for word space,
          transition probabilities between states for word space,
          emission probabilities for all states for word space )

        Sample Format (not complete):

        ( ['A1', ...],
          ['A1': prob_of_starting_in_A1, ...],
          {'A1': {'A1': prob_of_transition_from_A1_to_A1,
                  'A2': prob_of_transition_from_A1_to_A2}, ...},
          {'A1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['Y1', ...],
          ['Y1': prob_of_starting_in_Y1, ...],
          {'Y1': {'Y1': prob_of_transition_from_Y1_to_Y1,
                  'Y2': prob_of_transition_from_Y1_to_Y2}, ...},
          {'Y1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['Z1', ...],
          ['Z1': prob_of_starting_in_Z1, ...],
          {'Z1': {'Z1': prob_of_transition_from_Z1_to_Z1,
                  'Z2': prob_of_transition_from_Z1_to_Z2}, ...},
          {'Z1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['L1', ...]
          ['L1': prob_of_starting_in_L1, ...],
          {'L1': {'L1': prob_of_transition_from_L1_to_L1,
                  'L2': prob_of_transition_from_L1_to_L2}, ...},
          {'L1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['W1', ...]
          ['W1': prob_of_starting_in_W1, ...],
          {'W1': {'W1': prob_of_transition_from_W1_to_W1,
                  'W2': prob_of_transition_from_W1_to_W2}, ...},
          {'W1': [prob_of_observing_0, prob_of_observing_1], ...} )
        """

    """Letter A"""
    # expected states names for letter A
    a_states = ['A1', 'A2', 'A3', 'Aend']

    # prior probabilities for all states for letter A
    a_prior_probs = {'A1'  : 0.333,
           'A2'  : 0.0,
           'A3'  : 0.0,
           'Aend': 0.0}

    # transition probabilities between states for letter A
    a_transition_probs = {'A1'  : {'A1'  : 0.2,
                               'A2'  : 0.8,
                               'A3'  : 0.0,
                               'Aend': 0.0},
                      'A2'  : {'A1'  : 0.0,
                               'A2'  : 0.2,
                               'A3'  : 0.8,
                               'Aend': 0.0},
                      'A3'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.667,
                               'Aend': 0.333,
                               'L1':0.333,
                               'W1':0.333},
                      'Aend': {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.0,
                               'Aend': 1.0}}

    # emission probabilities for all states for letter A
    a_emission_probs = {'A1'  : [0.2, 0.8],
                    'A2'  : [0.8, 0.2],
                    'A3'  : [0.2, 0.8],
                    'Aend': [0.0, 0.0]}

    """Letter Y"""
    # expected states names for letter Y
    y_states = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Yend']

    y_prior_probs = { 'Y1': 0.333, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 }

    # transition probabilities between states for letter Y
    y_transition_probs = {
    'Y1':   { 'Y1': 0.667, 'Y2': 0.333,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 },
    'Y2':   { 'Y1': 0.0, 'Y2': 0.2,'Y3':0.8, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 },
    'Y3':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.2, 'Y4': 0.8, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 },
    'Y4':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.2, 'Y5': 0.8,'Y6':0.0, 'Y7':0.0, 'Yend':0.0 },
    'Y5':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.667,'Y6':0.333, 'Y7':0.0, 'Yend':0.0 },
    'Y6':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.2, 'Y7':0.8, 'Yend':0.0 },
    'Y7':   { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.667, 'Yend':0.111, 'L1':0.111, 'W1':0.111 },
    'Yend': { 'Y1': 0.0, 'Y2': 0.0,'Y3':0.0, 'Y4': 0.0, 'Y5': 0.0,'Y6':0.0, 'Y7':0.0, 'Yend':1.0 }
    }

    # emission probabilities for all states for letter Y
    y_emission_probs = {
    'Y1'   : [0.2,0.8],
    'Y2'   : [0.8,0.2],
    'Y3'   : [0.2,0.8],
    'Y4'   : [0.8,0.2],
    'Y5'   : [0.2,0.8],
    'Y6'   : [0.8,0.2],
    'Y7'   : [0.2,0.8],
    'Yend'  : [0,0]
    }

    """Letter Z"""
    # expected states names for letter Z
    z_states = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Zend']
    
    z_prior_probs = { 'Z1': 0.333, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 }

    # transition probabilities between states for letter Z
    z_transition_probs = {
    'Z1':   { 'Z1': 0.667, 'Z2': 0.333,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 },
    'Z2':   { 'Z1': 0.0, 'Z2': 0.2,'Z3':0.8, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 },
    'Z3':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.667, 'Z4': 0.333, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 },
    'Z4':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.2, 'Z5': 0.8,'Z6':0.0, 'Z7':0.0, 'Zend':0.0 },
    'Z5':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.2,'Z6':0.8, 'Z7':0.0, 'Zend':0.0 },
    'Z6':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.2, 'Z7':0.8, 'Zend':0.0 },
    'Z7':   { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.2, 'Zend':0.266 , 'L1':0.266, 'W1':0.266},
    'Zend': { 'Z1': 0.0, 'Z2': 0.0,'Z3':0.0, 'Z4': 0.0, 'Z5': 0.0,'Z6':0.0, 'Z7':0.0, 'Zend':1.0 }
    }

    # emission probabilities for all states for letter Z
    z_emission_probs = {
    'Z1'   : [0.2,0.8],
    'Z2'   : [0.8,0.2],
    'Z3'   : [0.2,0.8],
    'Z4'   : [0.8,0.2],
    'Z5'   : [0.2,0.8],
    'Z6'   : [0.8,0.2],
    'Z7'   : [0.2,0.8],
    'Zend'  : [0,0]
    }

    """Pause between letters"""
    # expected states names for letter pause
    letter_pause_states = ['L1']
    word_space_states = ['W1']
    # prior probabilities for all states for letter pause
    letter_pause_prior_probs = {'L1': 0.0}
    word_space_prior_probs = {'W1': 0.0}

    # transition probabilities between states for letter pause
    letter_pause_transition_probs = {'L1': {'L1': 0.667, 'A1': 0.111, 'Y1':0.111, 'Z1':0.111}}
    word_space_transition_probs = {'W1': {'W1': 0.857, 'A1': 0.0476, 'Y1':0.0476, 'Z1':0.0476}}
    # emission probabilities for all states for letter pause
    letter_pause_emission_probs = {'L1': [0.8, 0.2]}
    word_space_emission_probs = {'W1' : [0.8, 0.2]}

    return (a_states,
            a_prior_probs,
            a_transition_probs,
            a_emission_probs,
            y_states,
            y_prior_probs,
            y_transition_probs,
            y_emission_probs,
            z_states,
            z_prior_probs,
            z_transition_probs,
            z_emission_probs,
            letter_pause_states,
            letter_pause_prior_probs,
            letter_pause_transition_probs,
            letter_pause_emission_probs,
            word_space_states,
            word_space_prior_probs,
            word_space_transition_probs,
            word_space_emission_probs)


def quick_check():
    """Returns a few select values to check for accuracy.

    Returns:
        The following probabilities:
            ( prior probability of Z1,
              transition probability from Y7 to Y7,
              transition probability from Z3 to Z4,
              transition probability from W1 to W1,
              transition probability from L1 to Y1 )
    """

#    return (1.0, 0.667, 0.333, 0.857, 0.333)
#    raise NotImplemented()

    # prior probability for Z1
    prior_prob_Z1 = 0.333  # TODO

    # transition probability from Y7 to Y7
    transition_prob_Y7_Y7 = 0.667  # TODO

    # transition probability from Z3 to Z4
    transition_prob_Z3_Z4 = 0.333  # TODO

    # transition probability from W1 to W1
    transition_prob_W1_W1 = 0.857  # TODO

    # transition probability from L1 to Y1
    transition_prob_L1_Y1 = 0.111  # TODO

    return (prior_prob_Z1,
            transition_prob_Y7_Y7,
            transition_prob_Z3_Z4,
            transition_prob_W1_W1,
            transition_prob_L1_Y1)


def part_2_b(evidence_vector, states, prior_probs, transition_probs,
             emission_probs):
    """Decode the most likely string generated by the evidence vector.

    Note: prior, states, transition_probs, and emission_probs will now contain
    all the letters, pauses, and spaces from part_2_a.

    For example, prior is now:

    prior_probs = {'A1'   : 0.333,
                   'A2'   : 0.0,
                   'A3'   : 0.0,
                   'Aend' : 0.0,
                   'Y1'   : 0.333,
                   'Y2'   : 0.0,
                   .
                   .
                   .
                   'Z1'   : 0.333,
                   .
                   .
                   .
                   'L1'  : 0.0,
                   'W1   : 0.0}

    Expect the same type of combinations for all probability and state input
    arguments.

    Essentially, the built Viterbi Trellis will contain all states for A, Y, Z,
    letter pause, and word space.

    Args:
        evidence_vector (list(int)): List of 0s (Silence) or 1s (Dot/Dash).
            example: [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
        states (list(string)): List of all states.
            example: ['A1', 'A2', 'A3', 'Aend']
        prior_probs (dict): prior distribution for each state.
            example: {'A1'  : 1.0,
                      'A2'  : 0.0,
                      'A3'  : 0.0,
                      'Aend': 0.0}
        transition_probs (dict): dictionary representing transitions from
            each state to every other state, including self.
            example: {'A1'  : {'A1'  : 0.0,
                               'A2'  : 1.0,
                               'A3'  : 0.0,
                               'Aend': 0.0},
                      'A2'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 1.0,
                               'Aend': 0.0},
                      'A3'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.667,
                               'Aend': 0.333},
                      'Aend': {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.0,
                               'Aend': 1.0}}
        emission_probs (dict): dictionary of probabilities of outputs from
            each state.
            example: {'A1'  : [0.0, 1.0],
                      'A2'  : [1.0, 0.0],
                      'A3'  : [0.0, 1.0],
                      'Aend': [0.0, 0.0]}

    Returns:
        ( A string that best fits the evidence,
          probability of that string being correct as a float. )

        For example:
            an evidence vector of [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
            would return the String 'AA' with it's probability
    """
    sequence = []
    probability = 0.0
    
#    if evidence_vector == []:
#        return [],0.0
#    
#    if len(evidence_vector) == 1:
#        return [],0.0
    

    V = [{}]
    for state in states:
        V[0][state] = {"p": prior_probs[state]*emission_probs[state][evidence_vector[0]], "prev" : None}
        
    for t in range(1,len(evidence_vector)):
        V.append({})
#        print("xxxxxxxx")
#        print(t)
        for state in states[:-1]:
#            for prev_state in states[:-1]:
#                print("cccccccc")
#                
#                print(prev_state)
##                print(V)
#                print(V[t-1][prev_state]["p"])
#                print("------------------------")
#                print(state)
#                print(transition_probs[prev_state][state])
            def prob_func(prev_state, state):
                if state in transition_probs[prev_state].keys():
                    return transition_probs[prev_state][state]
                else:
                    return 0.0
            max_prev_prob = max(V[t-1][prev_state]["p"]*prob_func(prev_state, state) for prev_state in states[:-1])
            for prev_state in states[:-1]:
#                print("--------")
#                print(prev_state)
#                print(state)
                if V[t-1][prev_state]["p"]*prob_func(prev_state, state) == max_prev_prob:
                    max_prob = max_prev_prob*emission_probs[state][evidence_vector[t]]
                    print(state)
                    V[t][state] = {"p":max_prob, "prev":prev_state}
                    print(V[t][state])
                    break
    
    max_prob = max(prob["p"] for prob in V[-1].values())
    prev = None
    
    for state, value in V[-1].items():
        if value["p"] == max_prob:
            sequence.append(state)
            prev = state
            break
        
    for t in range(len(V)-2, -1, -1):
        sequence.insert(0, V[t+1][prev]["prev"])
        prev = V[t+1][prev]["prev"]
    seq_string = ''.join(sequence)
    seq_string = seq_string.replace("A1A2A3A3A3", "A")
    seq_string = seq_string.replace("L1L1L1", "")
    seq_string = seq_string.replace("L1", " ")
    seq_string = seq_string.replace("W1W1W1W1W1W1W1", " ")
    if(len(seq_string) > 2 ):
        while (seq_string[-2] == "W"):
            seq_string = seq_string[:-2]
    seq_string = seq_string.replace("Y1Y1Y1Y2Y3Y4Y5Y5Y5Y6Y7Y7Y7", "Y")
    seq_string = seq_string.replace("Z1Z1Z1Z2Z3Z3Z3Z4Z5Z6Z7", "Z")
    
    for i in range(3):
        if seq_string[-1] == " ":
            seq_string = seq_string[:-1]
    if(len(seq_string) > 2 ):
        if seq_string[-2] == 'A':
            replace = 'A'
        elif seq_string[-2] == 'Y':
            replace = 'Y'
        elif seq_string[-2] == 'Z':
            replace = 'Z'
        print("-------------------------")
        print(seq_string)
        if seq_string[-1] not in ['A', 'Y', 'Z']:
            print(seq_string[-1])
            while int(seq_string[-1]) > 1:
                rpl = 1
                seq_string = seq_string[:-2]
                if rpl == 1 :
                    
                    if seq_string[-2] == 'A':
                        seq_string = seq_string[:-2]
                    elif seq_string[-2] == 'Y':
                        seq_string = seq_string[:-2]
                    elif seq_string[-2] == 'Z':
                            seq_string = seq_string[:-2]
                if seq_string[-1] in ['A', 'Y', 'Z']:
                    break
            if seq_string[-1] == '1':
                seq_string = seq_string[:-2]
            seq_string = seq_string + replace
    return seq_string, max_prob

# Source: Viterbi Algorithm, wiki page