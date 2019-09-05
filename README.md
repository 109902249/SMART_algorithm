# MATLAB implementation of the SMART algorithm

0. Corresponding author: Qi Zhang, Department of Applied Mathematics and Statistics, Stony Brook University, Stony Brook, NY 11794-3600

1. The SMART algorithm by Qi Zhang and Jiaqiao Hu [1] is implemented for solving single-objective box-constrained expensive stochastic optimization problems.

2. SMART (surrogate- and model-based actor-critic like random technique) is a random search method for solving a class of simulation optimization problems with Lipschitz continuity properties. The algorithm samples candidate solutions from a parameterized probability distribution over the solution space, and estimates the performance of the sampled points through an asynchronous learning procedure based on the so-called shrinking ball method. A distinctive feature of the algorithm is that it fully retains the previous simulation information and incorporates an approximation architecture to allow the use of low-variance knowledge of the objective function in searching for improved solutions. Each step of the algorithm thus involves simultaneous adaptation of a parameterized distribution and an approximator of the objective function, which is akin to the actor-critic structure employed in reinforcement learning. Under appropriate conditions, we show that the algorithm converges globally when only a single simulation observation is collected at each iteration. Our numerical experiments indicate that the algorithm is promising and may significantly outperform some of the existing procedures in terms of both efficiency and reliability.

3. In this implementation, the algorithm samples candidate solutions from a sequence of independent multivariate normal distributions that recursively  approximiates the corresponding Boltzmann distributions [2].

4. In this implementation, the surrogate model is constructed by the radial basis function (RBF) method [3].

### Reference:
1. Qi Zhang and Jiaqiao Hu (2019): Actor-Critic Like Stochastic Adaptive Search for Continuous Simulation Optimization. Submitted to Operations Research, under review.
2. Jiaqiao Hu and Ping Hu (2011): Annealing adaptive search, cross-entropy, and stochastic approximation in global optimization. Naval Research Logistics 58(5):457-477.
3. Gutmann HM (2001): A radial basis function method for global optimization. Journal of Global Optimization 19:201-227.
