"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

#inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''


from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32, random
import numpy as np
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine, EnumerationEngine
import random

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []

    A_node = BayesNode(0, 2, name = "alarm")
    FA_node = BayesNode(1, 2, name = "faulty alarm")
    G_node = BayesNode(2, 2, name = "gauge")
    FG_node = BayesNode(3, 2, name = "faulty gauge")
    T_node = BayesNode(4, 2, name = "temperature")

    T_node.add_child(FG_node)
    T_node.add_child(G_node)
    FG_node.add_child(G_node)
    G_node.add_child(A_node)
    FA_node.add_child(A_node)

    G_node.add_parent(T_node)
    G_node.add_parent(FG_node)
    FG_node.add_parent(T_node)
    A_node.add_parent(G_node)
    A_node.add_parent(FA_node)

    nodes = [A_node, FA_node, G_node, FG_node, T_node]

    return BayesNet(nodes)
    raise NotImplementedError


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""
    A_node = bayes_net.get_node_by_name("alarm")
    FA_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    FG_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")

    #Temparature
    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([],[])
    T_distribution[index] = [0.8, 0.2]
    T_node.set_dist(T_distribution)

    #Faulty Alarm
    FA_distribution = DiscreteDistribution(FA_node)
    index = FA_distribution.generate_index([],[])
    FA_distribution[index] = [0.85, 0.15]
    FA_node.set_dist(FA_distribution)
    #Alarm
    dist = zeros([G_node.size(), FA_node.size() ,A_node.size()], dtype=float32)   #Note the order of G_node, A_node
    dist[0,0,:] = [0.90, 0.10]
    dist[0,1,:] = [0.55, 0.45]
    dist[1,0,:] = [0.10, 0.90]
    dist[1,1,:] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, FA_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

    #Faulty Gauge
    dist = zeros([T_node.size(), FG_node.size()], dtype=float32)   #Note the order of temp, Fg
    dist[0,:] = [0.95, 0.05]
    dist[1,:] = [0.20, 0.80]
    FG_distribution = ConditionalDiscreteDistribution(nodes=[T_node,FG_node], table=dist)
    FG_node.set_dist(FG_distribution)
    #Gauge
    dist = zeros([T_node.size(), FG_node.size() ,G_node.size()], dtype=float32)   #Note the order of G_node, A_node
    dist[0,0,:] = [0.95, 0.05]
    dist[0,1,:] = [0.20, 0.80]
    dist[1,0,:] = [0.05, 0.95]
    dist[1,1,:] = [0.80, 0.20]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, FG_node, G_node], table=dist)
    G_node.set_dist(G_distribution)

    return bayes_net
    raise NotImplementedError

#bayes_net = make_power_plant_net()
#bayes_net = set_probability(bayes_net)


def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal
    probability of the alarm
    ringing (T/F) in the
    power plant system."""
    A_node = bayes_net.get_node_by_name("alarm")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings], range(Q.nDims))
    alarm_prob = Q[index]

    return alarm_prob
    raise NotImplementedError

#print(get_alarm_prob(bayes_net, True))

def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge
    showing hot (T/F) in the
    power plant system."""
    G_node = bayes_net.get_node_by_name("gauge")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot], range(Q.nDims))
    gauge_prob = Q[index]

    return gauge_prob
    raise NotImplementedError
    
#print(get_gauge_prob(bayes_net, True))

def get_temperature_prob(bayes_net, temp_hot):
    """Calculate the conditional probability
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""

    T_node = bayes_net.get_node_by_name("temperature")
    A_node = bayes_net.get_node_by_name("alarm")
    FG_node = bayes_net.get_node_by_name("faulty gauge")
    FA_node = bayes_net.get_node_by_name("faulty alarm")

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[FG_node] = False
    engine.evidence[FA_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot], range(Q.nDims))
    temp_prob = Q[index]

    return temp_prob
    raise NotImplementedError

#print(get_temperature_prob(bayes_net, True))

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []

    A_node = BayesNode(0, 4, name = "A")
    B_node = BayesNode(1, 4, name = "B")
    C_node = BayesNode(2, 4, name = "C")

    AB_node = BayesNode(3, 3, name = "AvB")
    BC_node = BayesNode(4, 3, name = "BvC")
    CA_node = BayesNode(5, 3, name = "CvA")

    A_node.add_child(AB_node)
    A_node.add_child(CA_node)
    B_node.add_child(AB_node)
    B_node.add_child(BC_node)
    C_node.add_child(CA_node)
    C_node.add_child(BC_node)

    AB_node.add_parent(A_node)
    AB_node.add_parent(B_node)
    BC_node.add_parent(B_node)
    BC_node.add_parent(C_node)
    CA_node.add_parent(C_node)
    CA_node.add_parent(A_node)

    A_distribution = DiscreteDistribution(A_node)
    index = A_distribution.generate_index([],[])
    A_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    A_node.set_dist(A_distribution)

    B_distribution = DiscreteDistribution(B_node)
    index = B_distribution.generate_index([],[])
    B_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    B_node.set_dist(B_distribution)

    C_distribution = DiscreteDistribution(C_node)
    index = C_distribution.generate_index([],[])
    C_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    C_node.set_dist(C_distribution)

    dist = zeros([A_node.size(), B_node.size(), AB_node.size()], dtype=float32)
    dist[0, 0, :] = [0.10, 0.10, 0.80]
    dist[0, 1, :] = [0.20, 0.60, 0.20]
    dist[0, 2, :] = [0.15, 0.75, 0.10]
    dist[0, 3, :] = [0.05, 0.90, 0.05]

    dist[1, 0, :] = [0.60, 0.20, 0.20]
    dist[1, 1, :] = [0.10, 0.10, 0.80]
    dist[1, 2, :] = [0.20, 0.60, 0.20]
    dist[1, 3, :] = [0.15, 0.75, 0.10]

    dist[2, 0, :] = [0.75, 0.15, 0.10]
    dist[2, 1, :] = [0.60, 0.20, 0.20]
    dist[2, 2, :] = [0.10, 0.10, 0.80]
    dist[2, 3, :] = [0.20, 0.60, 0.20]

    dist[3, 0, :] = [0.90, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.10]
    dist[3, 2, :] = [0.60, 0.20, 0.20]
    dist[3, 3, :] = [0.10, 0.10, 0.80]

    AB_distribution = ConditionalDiscreteDistribution(nodes=[A_node, B_node, AB_node], table = dist)
    AB_node.set_dist(AB_distribution)
    BC_distribution = ConditionalDiscreteDistribution(nodes=[B_node, C_node, BC_node], table = dist)
    BC_node.set_dist(BC_distribution)
    CA_distribution = ConditionalDiscreteDistribution(nodes=[C_node, A_node, CA_node], table = dist)
    CA_node.set_dist(CA_distribution)

    nodes = [A_node, B_node, C_node, AB_node, BC_node, CA_node]

    return BayesNet(nodes)
    raise NotImplementedError



def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C.
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]

    AB_node = bayes_net.get_node_by_name("AvB")
    BC_node = bayes_net.get_node_by_name("BvC")
    CA_node = bayes_net.get_node_by_name("CvA")

    engine=EnumerationEngine(bayes_net)
    engine.evidence[AB_node] = 0
    engine.evidence[CA_node] = 2
    Q = engine.marginal(BC_node)[0]
    index = Q.generate_index([], range(Q.nDims))
    B, C, tie = Q[index]
    posterior = [B, C, tie]

    return posterior
    raise NotImplementedError


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm
    given a Bayesian network and an initial state value.

    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)

    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.
    """
    sample = tuple(initial_state)

    if (initial_state == None) or (initial_state == []):
        initial_state = [0,0,0,0,0,2]

    # setting up evidence variables
    initial_state[3] = 0
    initial_state[5] = 2

    A_node = bayes_net.get_node_by_name("A")
    AB_node = bayes_net.get_node_by_name("AvB")
    dist = A_node.dist.table
    match_dist = AB_node.dist.table
    #randomly selecting non-evidence variables
    random_choice = random.choice([0, 1, 2, 4])
    #print("random choice")
    #print(random_choice)
    if random_choice > 2:
        left_parent = initial_state[random_choice - 3]
        right_parent = initial_state[(random_choice - 3 + 1)%3]
        prob_dist = match_dist[left_parent, right_parent, :]
        update_gibbs = np.random.choice([0,1,2], p = prob_dist)
        #print("update - 1")
        #print(update_gibbs)
    else: # sampling for team probabilities using the Markov Blanket concept from text book
        skills = len(dist)
        prob_sampling = [0]*skills
        left_team = initial_state[(random_choice - 1) % 3]
        right_team = initial_state[(random_choice + 1) % 3]
        #print(random_choice)
        #print(left_team + 3)
        left_child = initial_state[((random_choice - 1) % 3) + 3]
        right_child = initial_state[random_choice + 3]

        for skill_level in range(skills):
            a = match_dist[skill_level, right_team, right_child]
            b = match_dist[left_team, skill_level, left_child]
            prob_sampling[skill_level] = dist[skill_level] * a * b
        normalized = [float(i)/sum(prob_sampling) for i in prob_sampling]
        #print(normalized)
        #print(sum(normalized))
        prob_sampling = np.array(prob_sampling)
        normalized = prob_sampling/prob_sampling.sum()
        update_gibbs = np.random.choice([0, 1, 2, 3], p = normalized)
        #print("update - 2")
        #print(update_gibbs)

    initial_state[random_choice] = update_gibbs
    sample = tuple(initial_state)
    #print(sample)
    return sample
    raise NotImplementedError


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """

    if (initial_state == None) or (initial_state == []):
        initial_state = [0,0,0,0,0,2]

    # setting up evidence variables
    initial_state[3] = 0
    initial_state[5] = 2

    A_node = bayes_net.get_node_by_name("A")
    AB_node = bayes_net.get_node_by_name("AvB")
    A_dist = A_node.dist.table
    AB_dist = AB_node.dist.table

    B_node = bayes_net.get_node_by_name("B")
    BC_node = bayes_net.get_node_by_name("BvC")
    B_dist = B_node.dist.table
    BC_dist = BC_node.dist.table

    C_node = bayes_net.get_node_by_name("C")
    CA_node = bayes_net.get_node_by_name("CvA")
    C_dist = C_node.dist.table
    CA_dist = CA_node.dist.table

    skills = [0, 1, 2, 3]
    results = [0, 1, 2]
    non_evidence = [0, 1, 2, 4]
    update_state = list(initial_state)

    for x in non_evidence:
        if x < 3: update_state[x] = random.choice(skills)
    else:
        update_state[x] = random.choice(results)

    A, B, C, AB, BC, CA = initial_state
    pi_initial_state = A_dist[A] * B_dist[B] * C_dist[C] * AB_dist[A][B][AB] * BC_dist[B][C][BC] * CA_dist[C][A][CA]
    A, B, C, AB, BC, CA = update_state
    pi_update_state = A_dist[A] * B_dist[B] * C_dist[C] * AB_dist[A][B][AB] * BC_dist[B][C][BC] * CA_dist[C][A][CA]
    # Acceptence Test
    acceptance_ratio = pi_update_state/pi_initial_state
    alpha = min(1, acceptance_ratio)
    u = random.uniform(0,1)

    if u < alpha:
        sample = tuple(update_state)
    else:
        sample = tuple(initial_state)
    return sample
    raise NotImplementedError


def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    N = 100
    N_tracker = 0
    delta = 0.00001
    max_iter = 0

    if (initial_state == None) or (initial_state == []):
      initial_state = [0, 0, 0, 0, 0, 2]

    cur_state = list(initial_state)
    result_counts = [0, 0, 0]
    while max_iter < 10000000:
      max_iter += 1
      gibbs_update = Gibbs_sampler(bayes_net, cur_state)
      Gibbs_count += 1
      cur_state = list(gibbs_update)
      Gibbs_prev_convergence = Gibbs_convergence
      #print(cur_state)
      if cur_state[4] == 0: # if B wins
        result_counts[0] += 1
      elif cur_state[4] == 1:
        result_counts[1] += 1 # if C wins
      else:
        result_counts[2] += 1

      Gibbs_convergence = [float(i)/sum(result_counts) for i in result_counts]
      # Delta Check
      differ = zip(Gibbs_convergence,Gibbs_prev_convergence)
      differ = [abs(x - y) for x,y in differ]
      #diff = abs(Gibbs_convergence - Gibbs_prev_convergence)
      if (all(i <= delta for i in differ)):
        N_tracker += 1
      else:
        N_tracker = 0

      if (N_tracker >= N) and (Gibbs_count > 200000):
        break


    max_iter = 0
    cur_state = list(initial_state)
    result_counts = [0, 0, 0]
    N_tracker = 0
    while max_iter < 1000000:
      max_iter += 1
      MH_update = MH_sampler(bayes_net, cur_state)
      MH_count += 1
      if (cur_state == list(MH_update)): MH_rejection_count += 1
      cur_state = list(MH_update)
      MH_prev_convergence = MH_convergence

      if cur_state[4] == 0: # if B wins
        result_counts[0] += 1
      elif cur_state[4] == 1:
        result_counts[1] += 1 # if C wins
      else:
        result_counts[2] += 1

      MH_convergence = [float(i)/sum(result_counts) for i in result_counts]
      # Delta Check
      differ = zip(MH_convergence,MH_prev_convergence)
      differ = [abs(x - y) for x,y in differ]
      if (all(i <= delta for i in differ)):
        N_tracker += 1
      else:
        N_tracker = 0

      if (N_tracker >= N) and (MH_count > 200000):
        break

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count
    raise NotImplementedError


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    name = "Rajesh Pothamsetty"
    return name
    raise NotImplementedError


#print(compare_sampling(get_game_network(), [], 0.0001))