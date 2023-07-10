import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from tikzplotlib import save


def maxmin(A, solver="glpk"):

    num_vars = len(A)
    # minimize matrix c
    c = [-1] + [0 for i in range(num_vars)]
    c = np.array(c, dtype="float")
    c = matrix(c)
    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T  # reformat each variable is in a row
    G *= -1  # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1])  # > 0 constraint for all vars
    new_col = [1 for i in range(num_vars)] + [0 for i in range(num_vars)]
    G = np.insert(G, 0, new_col, axis=1)  # insert utility column
    G = matrix(G)
    h = ([0 for i in range(num_vars)] +
         [0 for i in range(num_vars)])
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [0] + [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver, options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    return np.array(sol["x"]), sol


class LEWIS():
    def __init__(self, na_p1, payoff_matrix, discount_factor, ep=0.1):
        self.na_p1 = na_p1  # Number of actions of p1
        self.payoff_matrix = payoff_matrix
        self.discount_factor = discount_factor  # Discount factor to use in the game

        # Values needed for Q-block
        self.Q = None
        self.alpha = 0.5  # Updating value

        # Initialize the strategy blocks
        self.strategy_blocks = [self.q_strategy_block]
        self.nsb = len(self.strategy_blocks)  # Number of strategy blocks used
        # Other parameters
        self.ep = ep  # For epsilon equilibria
        self.t = 0  # Time index
        self.r = None #  Discounted past rewards

        # Maxmin values
        sol, _ = maxmin(self.payoff_matrix)
        self.a_mm = sol[1:].reshape(self.na_p1)  # Minimax action
        self.a_mm = np.clip(self.a_mm, 0, 1) # Careful: sometimes the linear solver provides values like 0 negative...
        self.a_mm = self.a_mm / sum(self.a_mm)
        self.v_mm = sol[0][0]  # Minimax value

    def q_strategy_block(self):
        # always recommend the max Q-value
        r_est = np.amax(self.Q)  # Max value
        action = np.argwhere(self.Q.tolist() == r_est)
        if action.size == 1:
            action = np.squeeze(action)
        else:
            action = np.random.choice(np.squeeze(action))  # If there are more than 1 action giving the max, select randomly
        r_worse = np.amin(self.payoff_matrix[action, :])

        return action, r_worse, r_est

    def reset(self):
        self.t = 0  # Time index
        self.r = []  # Discounted past rewards
        self.Q = self.payoff_matrix[np.arange(self.na_p1), np.argmax(self.payoff_matrix, axis=1)].astype(float)

    def get_BR(self, prob_opp):
        vals = self.payoff_matrix[:, :].dot(prob_opp)  # Obtain matrix product for P1
        return np.argmax(vals), np.amax(vals)

    def get_action(self):
        # Obtain actions and rewards for each strategy block
        actions = []
        r_w = []
        r_b = []
        for sb in range(self.nsb):
            a, rw, rb = self.strategy_blocks[sb]()  # Execute the strategy block
            actions.append(a)
            r_w.append(rw)
            r_b.append(rb)
        # Obtain the security conditions
        V_clb = (1 - self.discount_factor) * sum(self.r) + self.discount_factor ** self.t * self.v_mm  # Cumulative low bound payoff
        V_tlb = self.v_mm  # Total low bound value
        V_clb = V_tlb
        V_cwb1 = np.zeros(self.nsb)  # Worst value by following the strategy block once
        for sb in range(self.nsb):
            V_cwb1[sb] = (1 - self.discount_factor) * sum(self.r) \
                         + (self.discount_factor ** self.t - self.discount_factor ** (self.t + 1)) * r_w[sb] \
                         + self.discount_factor ** (self.t + 1) * self.v_mm
        # Check the security conditions
        valid_strategies = []
        for sb in range(self.nsb):
            if V_cwb1[sb] >= V_clb - self.ep and V_cwb1[sb] >= V_tlb - self.ep:
                valid_strategies.append(sb)
        if len(valid_strategies) == 0:
            return np.argmax(np.random.multinomial(1,self.a_mm))  # Follow the MS
        else: # find the valid strategy that yields the maximum value
            valid_values = [r_b[i] for i in valid_strategies]
            a = actions[r_b.index(np.amax(valid_values))]
            return np.argmax(np.random.multinomial(1,np.array([1 - a, a])))  # Follow the strategy provided

    def update(self, a1, r1):
        self.Q[a1] = (1 - self.alpha) * self.Q[a1] + self.alpha * r1
        self.r.append(self.discount_factor ** self.t * r1)  # Store the stage reward
        self.t += 1


if __name__=="__main__":

    lam = 628 / 960.0
    u1 = np.array([[0, -1], [1, -lam]])  # Player I
    u2 = np.array([[0, 1], [-1, -lam]])  # Player P

    delta1v = [0.6, 0.7, 0.8, 0.85, 0.9]
    delta2 = 0.95

    payoff_data = np.zeros((len(delta1v), 2, 3))  # Delta x player x case (baseline, nash, lewis), when ep = 0.1

    for delta1 in delta1v:

        assert 0 < delta1 < delta2 < 1

        # Parameters of baseline
        e = (2*delta1-1) / (delta1-lam/(lam+1))-3/2
        k = np.floor(np.log((1-lam)/(3/2+e))/np.log(delta2))

        cond1 = delta1 > 0.99*(1-lam*(e+1/2))/((1+lam)*(1/2-e))  # The 0.99 is to account for numerical errors
        cond2 = delta2 ** k > (1-lam)/(e+3/2)
        cond3 = delta2 > (1/2-e)/(1+lam)
        cond4 = 0<=e<=0.5

        print(cond1, cond2, cond3, cond4)

        tmax = 100
        reps = 30  # Repetitions for each game!

        # Simulate
        epv = [0, 0.01, 0.1, 1]
        data = np.zeros((4, reps, len(epv)))
        data_tx = np.zeros((4, reps, len(epv)))
        data_cols = np.zeros((2, reps, len(epv)))
        for ep in epv:
            for rep in range(reps):
                if cond1 and cond2 and cond3 and cond4:  # We can apply baseline
                    rwd_bas = np.zeros((2, tmax))
                    for t in range(tmax):
                        if t < k: # First phase: benefit player I
                            action1 = 1
                            action2 = 0
                        else:
                            action1 = 0
                            action2 = np.argmax(np.random.multinomial(1,np.array([1/2-e, 1/2+e])))
                        rwd_bas[0, t] = u1[action1, action2]
                        rwd_bas[1, t] = u2[action1, action2]
                        if action1 > 0.5:
                            data_tx[0, rep, epv.index(ep)] += 1
                        if action2 > 0.5:
                            data_tx[1, rep, epv.index(ep)] += 1
                        if action1 > 0.5 and action2 > 0.5:
                            data_cols[0, rep, epv.index(ep)] += 1
                else:
                    rwd_bas = -lam*np.ones((2, tmax))

                players = [LEWIS(2, u1, delta1, ep=ep), LEWIS(2, u2.T, delta2, ep=ep)]
                for p in players:
                    p.reset()
                rwd_lew = np.zeros((2, tmax))
                for t in range(tmax):
                    action1 = players[0].get_action()
                    action2 = players[1].get_action()
                    rwd_lew[0, t] = u1[action1, action2]
                    rwd_lew[1, t] = u2[action1, action2]
                    players[0].update(action1, rwd_lew[0, t])
                    players[1].update(action2, rwd_lew[1, t])

                    if action1 > 0.5:
                        data_tx[2, rep, epv.index(ep)] += 1
                    if action2 > 0.5:
                        data_tx[3, rep, epv.index(ep)] += 1
                    if action1 > 0.5 and action2 > 0.5:
                        data_cols[1, rep, epv.index(ep)] += 1

                crw_bas_1 = (1-delta1)*np.sum(np.power(delta1, np.arange(tmax))*rwd_bas[0, :])
                crw_bas_2 = (1-delta2)*np.sum(np.power(delta2, np.arange(tmax)) * rwd_bas[1, :])
                crw_lew_1 = (1 - delta1) * np.sum(np.power(delta1, np.arange(tmax)) * rwd_lew[0, :])
                crw_lew_2 = (1 - delta2) * np.sum(np.power(delta2, np.arange(tmax)) * rwd_lew[1, :])
                data[:, rep, epv.index(ep)] = np.array([crw_bas_1, crw_bas_2, crw_lew_1, crw_lew_2])

                if np.square(ep - 0.1) < 1e-6:
                    payoff_data[delta1v.index(delta1), 0, :] = np.array([crw_bas_1, -lam, crw_lew_1])
                    payoff_data[delta1v.index(delta1), 1, :] = np.array([crw_bas_2, -lam, crw_lew_2])

        plt.plot(epv, np.mean(np.squeeze(data).T, axis=1))
        plt.plot(epv, [-lam] * len(epv), 'c')
        plt.title('delta = ' + str(delta1))
        save('wban_' + str(delta1) + '.tex')
        plt.show()
        '''
        plt.plot(np.mean(np.squeeze(data_tx).T, axis=1))
        plt.show()
        plt.plot(np.mean(np.squeeze(data_cols).T, axis=1))
        plt.show()
        '''
    plt.plot(delta1v, payoff_data[:, 0, 0], 'ro-')
    plt.plot(delta1v, payoff_data[:, 1, 0], 'rs-')
    plt.plot(delta1v, payoff_data[:, 0, 1], 'bo-')
    plt.plot(delta1v, payoff_data[:, 1, 1], 'bs-')
    plt.plot(delta1v, payoff_data[:, 0, 2], 'ko-')
    plt.plot(delta1v, payoff_data[:, 1, 2], 'ks-')
    save('wban_comp.tex')
    plt.show()
    print('Done')