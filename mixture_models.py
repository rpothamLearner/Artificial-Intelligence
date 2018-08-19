from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)


def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    flattened = flatten_image_matrix(image_values)
    count = 0
    # M step, random prototype vector
    if initial_means == None:
      initial_means_idx = np.random.choice(flattened.shape[0], k, replace = False)
      mu_k = flattened[initial_means_idx]
    else:
      mu_k = initial_means
    
    while count < 1:
        # E step - updating r_nk
        squared_dist = np.array([np.square(flattened - mu_k[i]).sum(axis = 1) for i in range(mu_k.shape[0])])
        r_nk = np.argmin(squared_dist, axis = 0)
        # M step - update mu_k
        clusters = np.array([flattened[(r_nk == i)]  for i in range(k)])
        mu_k_prev = mu_k
        mu_k = np.array([clusters[i].mean(axis = 0) for i in range(k)])
        # Convergence test
        if np.array_equal(mu_k_prev, mu_k):
          count += 1

    image_matrix = np.array([mu_k[k] for k in r_nk])
    updated_image_values = unflatten_image_matrix(image_matrix, image_values.shape[1])
    return updated_image_values
    raise NotImplementedError()


def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            #print(num_components)
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        def component_joint_prob(i):
          a = 0.5*np.log(2*np.pi*self.variances[i])
          b = ((val - self.means[i])**2)/(2*self.variances[i])
          c = np.log(self.mixing_coefficients[i])
          return -a - b + c
        return logsumexp([component_joint_prob(i) for i in range(self.num_components)])
        raise NotImplementedError()

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        flattened = flatten_image_matrix(self.image_matrix)
        rand_idx = np.random.choice(flattened.shape[0], self.num_components, replace = False)
        self.means = flattened[rand_idx]  #.tolist()
        self.variances = np.ones(self.num_components)
        self.mixing_coefficients = np.full(self.num_components, 1/self.num_components)
#        raise NotImplementedError()

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
        flattened = flatten_image_matrix(self.image_matrix)# n X 1
        conv_ctr = 0
        #print(self.means)
        #print(self.variances)
        def component_joint_prob(i):
          a = 0.5*np.log(2*np.pi*self.variances[i])
          b = (np.square(flattened - self.means[i]))/(2*self.variances[i])
          c = np.log(self.mixing_coefficients[i])
          return -a - b + c

        while conv_ctr < 10:
          prev_likelihood = self.likelihood() #+ 150
          prev_variables = np.array([np.array(self.means).tolist(), np.array(self.variances).tolist(), np.array(self.mixing_coefficients).tolist()])
          # formulae Bishop equation 9.23
          z = np.array([component_joint_prob(i) for i in range(self.num_components)]) # 3 X n
          gamma_nk_denom = logsumexp(z, axis = 0)
          for k in range(self.num_components):
            # E step
            gamma_nk_nomin = z[k]
            #print(z[k])
            gamma_nk = np.exp(gamma_nk_nomin - gamma_nk_denom)
            N_k = gamma_nk.sum()
            #print("array_constant_N_k")
            #print(N_k)
            # M step
            mu_k = np.multiply(gamma_nk, flattened).sum()/N_k
            x_muk = flattened - mu_k
            sigma_k = np.multiply(gamma_nk, np.square(x_muk)).sum()/N_k #covar matrix
            #print(gamma_nk.shape)
            #print("diff shape")
            #print((flattened - mu_k).shape)
            pi_k = N_k/flattened.shape[0]
            # update
            self.means[k] = mu_k
            self.variances[k] = sigma_k
            self.mixing_coefficients[k] = pi_k

          new_likelihood = self.likelihood() #+ 150
          new_variables = np.array([np.array(self.means).tolist(), np.array(self.variances).tolist(), np.array(self.mixing_coefficients).tolist()])
          if convergence_function == default_convergence:
            conv_ctr, converge_bool = convergence_function(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10)
          else:
            conv_ctr, converge_bool = convergence_function(prev_variables, new_variables, conv_ctr, conv_ctr_cap=10)
# raise NotImplementedError()

    def segment(self):

        flattened = flatten_image_matrix(self.image_matrix)# n X 1

        def component_joint_prob(i):
          a = 0.5*np.log(2*np.pi*self.variances[i])
          b = (np.square(flattened - self.means[i]))/(2*self.variances[i])
          c = np.log(self.mixing_coefficients[i])
          return -a - b + c

        z = np.array([component_joint_prob(i) for i in range(self.num_components)]) # 3 X n
        gamma_nk_denom = logsumexp(z, axis = 0)
        gamma_component = []
        for k in range(self.num_components):
            gamma_nk_nomin = z[k]
            gamma_nk = np.exp(gamma_nk_nomin - gamma_nk_denom)
            gamma_component.append(gamma_nk)
        idx = np.argmax(np.array(gamma_component), axis = 0)
        #print(idx)
        image_matrix = np.array([self.means[i] for i in idx.flatten()])
        update_image_values = unflatten_image_matrix(image_matrix, self.image_matrix.shape[1])
        return update_image_values

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        flattened = flatten_image_matrix(self.image_matrix) # n X 1
        def component_joint_prob(i):
          a = 0.5*np.log(2*np.pi*self.variances[i])
          b = ((flattened - self.means[i])**2)/(2*self.variances[i])
          c = np.log(self.mixing_coefficients[i])
          return -a - b + c
        z = np.array([component_joint_prob(i) for i in range(self.num_components)]) # 3 X n
        likelihood = logsumexp(z, axis = 0).sum()
        return likelihood
        raise NotImplementedError()

    def best_segment(self, iters):
        
        segment = None
        max_likelihood = -float('inf')

        for i in range(iters):
            self.train_model()
            cur_likelihood = self.likelihood()
            if max_likelihood < cur_likelihood:
                segment = self.segment()
                max_likelihood = cur_likelihood
        return segment.reshape((segment.shape[0], segment.shape[1]))
        raise NotImplementedError()


class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
        flattened = flatten_image_matrix(self.image_matrix)
        count = 0
        k = self.num_components
    # M step, random prototype vector
        initial_means_idx = np.random.choice(flattened.shape[0], k, replace = False)
        mu_k = flattened[initial_means_idx]

        while count < 1:
        # E step - updating r_nk
          squared_dist = np.array([np.square(flattened - mu_k[i]).sum(axis = 1) for i in range(mu_k.shape[0])])
          r_nk = np.argmin(squared_dist, axis = 0)
          # M step - update mu_k
          clusters = np.array([flattened[r_nk == i]  for i in range(k)])
          mu_k_prev = mu_k
          mu_k = np.array([clusters[i].mean(axis = 0) for i in range(k)])
          # Convergence test
          if np.array_equal(mu_k_prev, mu_k):
            count += 1
        #print(1/self.num_components)
        mixing_coefficient = 1.0/self.num_components
        self.means = mu_k.tolist()
        self.variances = [1]*self.num_components
        self.mixing_coefficients = [mixing_coefficient]*self.num_components
#        raise NotImplementedError()


def new_convergence_function(previous_variables, new_variables, conv_ctr, conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]] containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]] containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    def great_less_than(x,y):
      return (abs(x) * 0.98 < abs(y) < abs(x) * 1.02)
    summ = []
    n = len(previous_variables[0])
    for i in range(n):
      summ.append([great_less_than(x,y) for x,y in zip(previous_variables[i], new_variables[i])])
    
    if np.array(summ).sum() == np.square(n):
      conv_ctr += 1
    else:
      conv_ctr = 0
    print(np.array(summ).sum())
    return conv_ctr, conv_ctr > conv_ctr_cap

class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        GaussianMixtureModel.train_model(self, convergence_function)
#        raise NotImplementedError()


def bayes_info_criterion(gmm):
    n = gmm.image_matrix.shape[0]*gmm.image_matrix.shape[1]
    BIC = np.log(n)*gmm.num_components * 3 - 2*gmm.likelihood()
    return BIC
    raise NotImplementedError()


def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
      """
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]

    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    bic= []
    likelihood = []
    for k in range(2,8):
      #print(k)
      gmm = GaussianMixtureModel(image_matrix, k)
      gmm.initialize_training()
      gmm.means = np.copy(comp_means[k-2])
      gmm.train_model()
      bic.append(bayes_info_criterion(gmm))
      likelihood.append(gmm.likelihood())
    print(likelihood)
    min_BIC_model = np.argmin(bic) + 2
    max_likelihood_model = np.argmax(likelihood) + 2
    return min_BIC_model, max_likelihood_model
    raise NotImplementedError()


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
#    raise NotImplementedError()
    bic = 7
    likelihood = 7
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs

def return_your_name():
    # return your name
    name = "Rajesh Pothamsetty"
    return name
    raise NotImplemented()

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.

    returns:
    dists = numpy array of float
    """
    # TODO: fill in the bonus function
    # REMOVE THE LINE BELOW IF ATTEMPTING BONUS
    return dists
