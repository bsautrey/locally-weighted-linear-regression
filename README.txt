Written by Ben Autrey: https://github.com/bsautrey

---Overview---

Implement locally weighted linear regression from Andrew Ng's CS229 course: http://cs229.stanford.edu/notes/cs229-notes1.pdf. Batch gradient descent is used to learn the parameters, i.e. minimize the cost function.

alpha - The learning rate.
dampen - Factor by which alpha is dampened on each iteration. Default is no dampening, i.e. dampen = 1.0
tol - The stopping criteria
theta - The parameters to be learned.
weights - The weighting used for each data point, for each local regression.
index - A point, 0 - m, where the local regression is to be evaluated.
percents - A list containing the iteration number and the percent change in theta, from one iteration to the next.

---Requirements---

* numpy: https://docs.scipy.org/doc/numpy/user/install.html
* matplotlib: https://matplotlib.org/users/installing.html

---Example---

1) Change dir to where LWR.py is.

2) Run this in a python terminal:

from LWR import LWR
lwr = LWR()
lwr.generate_example(500)

OR

See the function generate_example() in LWR.py.