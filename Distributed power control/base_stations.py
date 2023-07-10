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
            action = int(np.squeeze(action))
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
            if a <= 1:
                return np.argmax(np.random.multinomial(1,np.array([1 - a, a])))  # Follow the strategy provided
            else:
                return a

    def update(self, a1, r1):
        self.Q[a1] = (1 - self.alpha) * self.Q[a1] + self.alpha * r1
        self.r.append(self.discount_factor ** self.t * r1)  # Store the stage reward
        self.t += 1

def learn(actions, u1, u2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma):
    a1m = actions[-1]
    a2m = actions[-1]
    a1v = []
    a2v = []
    p1_content = False
    p2_content = False
    c = 3
    max_payoff1 = np.amax(u1)
    min_payoff1 = np.amin(u1)
    max_payoff2 = np.amax(u2)
    min_payoff2 = np.amin(u2)

    for i in range(500):
        t = i + 1
        epsilon = 1 / np.sqrt(t)
        if p1_content:
            prob_vector = [(epsilon ** c) / (len(actions) - 1)] * len(actions)
            prob_vector[actions.index(a1m)] = 1 - epsilon ** c
            aindex = np.argmax(np.random.multinomial(1, np.array(prob_vector)))
            a1 = actions[aindex]
        else:
            prob_vector = [1 / len(actions)] * len(actions)
            aindex = np.argmax(np.random.multinomial(1, np.array(prob_vector)))
            a1 = actions[aindex]
        if p2_content:
            prob_vector = [(epsilon ** c) / (len(actions) - 1)] * len(actions)
            prob_vector[actions.index(a2m)] = 1 - epsilon ** c
            aindex = np.argmax(np.random.multinomial(1, np.array(prob_vector)))
            a2 = actions[aindex]
        else:
            prob_vector = [1 / len(actions)] * len(actions)
            aindex = np.argmax(np.random.multinomial(1, np.array(prob_vector)))
            a2 = actions[aindex]
        u1n, u2n = payoff(a1, a2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)
        u1n = (u1n - min_payoff1) / (max_payoff1 - min_payoff1)
        u2n = (u2n - min_payoff2) / (max_payoff2 - min_payoff2)
        if (p1_content and a1 != a1m) or not p1_content:
            a1m = a1
            aindex = np.argmax(np.random.multinomial(1, np.array([epsilon ** (1 - u1n), 1 - epsilon ** (1 - u1n)])))
            if aindex < 0.5:
                p1_content = True
            else:
                p1_content = False
        if (p2_content and a2 != a2m) or not p2_content:
            a2m = a2
            aindex = np.argmax(np.random.multinomial(1, np.array([epsilon ** (1 - u2n), 1 - epsilon ** (1 - u2n)])))
            if aindex < 0.5:
                p2_content = True
            else:
                p2_content = False
        a1v.append(a1m)
        a2v.append(a2m)
    return a1m, a2m, a1v, a2v


def simulate(tmax, actions, a1m, a2m, delta, ep):
    # Baseline simulation: no deviation
    rwd_bas = np.zeros((2, tmax))

    action1 = actions.index(a1m)
    action2 = actions.index(a2m)

    for t in range(tmax):
        rwd_bas[0, t] = u1[action1, action2]
        rwd_bas[1, t] = u2[action1, action2]

    # Baseline simulation with deviation at t=tdev
    rwd_bas_dev = np.zeros((2, tmax))
    tdev = 1

    for t in range(tmax):
        if t < tdev:
            action1 = actions.index(a1m)
            action2 = actions.index(a2m)
        elif t == tdev:
            action1 = actions.index(a1m)
            action2 = int(np.argmax(u2[action1, :]))
        else:  # Grim
            action1 = action2 = len(actions) - 1  # Grim action: the last one (max power)
        rwd_bas_dev[0, t] = u1[action1, action2]
        rwd_bas_dev[1, t] = u2[action1, action2]

    players = [LEWIS(len(actions), u1, delta, ep=ep), LEWIS(len(actions), u2.T, delta, ep=ep)]
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

    return rwd_bas, rwd_bas_dev, rwd_lew


def distance(p1, p2):  # Returns distance between two positions
    return np.sqrt(np.sum(np.square(p1-p2)))


def att(p1, p2, alpha=4):  # REturns the attenuation between two positions
    return distance(p1, p2) ** (-alpha)


def sinr(p1, p2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma):
    sinr1ul = p_ul * att(pos_sb1, pos_u1) / (
                n0 + p_ul * att(pos_sb1, pos_u2) + gamma * (p1 + p2 * att(pos_sb1, pos_sb2)))
    sinr1dl = p1 * att(pos_sb1, pos_u1) / (
                n0 + p2 * att(pos_u1, pos_sb2) + gamma * (p_ul + p_ul * att(pos_u2, pos_u1)))
    sinr2ul = p_ul * att(pos_sb2, pos_u2) / (
                n0 + p_ul * att(pos_sb2, pos_u1) + gamma * (p2 + p1 * att(pos_sb1, pos_sb2)))
    sinr2dl = p2 * att(pos_sb2, pos_u2) / (
                n0 + p1 * att(pos_u2, pos_sb1) + gamma * (p_ul + p_ul * att(pos_u2, pos_u1)))
    return sinr1ul, sinr1dl, sinr2ul, sinr2dl


def payoff(p1, p2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma):
    sinr1ul, sinr1dl, sinr2ul, sinr2dl = sinr(p1, p2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)
    return np.log(sinr1ul) + np.log(sinr1dl), np.log(sinr2ul) + np.log(sinr2dl)


if __name__=="__main__":

    np.random.seed(10)  # For repeatability
    N = 2  # Number of small cells
    pos_sb1 = np.array([10, 10])  # Position of small base 1
    pos_sb2 = np.array([0, 0])  # Position of small base 2
    pos_u1 = np.array([1, 8])  # Position of user of sb1
    pos_u2 = np.array([5, 5])  # Position of user of sb2

    n0 = 0.001  # Thermal noise
    p_ul = 5  # Constant uplink power!
    gamma = 0.001  # Cochannel interference factor

    actions = [5, 10, 15, 20, 25, 30]  # Power levels (actions!)

    pv = []
    u1 = np.zeros((len(actions), len(actions)))
    u2 = np.zeros((len(actions), len(actions)))
    for p1 in actions:
        for p2 in actions:
            pv.append(payoff(p1, p2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma))
            u1[actions.index(p1), actions.index(p2)] = pv[-1][0]
            u2[actions.index(p1), actions.index(p2)] = pv[-1][1]

    a1m, a2m, a1v, a2v = learn(actions, u1, u2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)  # Learning phase

    pi1, pi2 = payoff(np.array(a1v), np.array(a2v), pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)
    p1n, p2n = payoff(actions[-1], actions[-1], pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)  # Nash payoff
    plt.plot(pi1, 'r')
    plt.plot(pi2, 'b')
    plt.plot([p1n] * len(pi1), 'ro')
    plt.plot([p2n] * len(pi2), 'bo')
    plt.title('Learning phase')
    save('base_learning.tex')
    plt.show()


    # SIMULATION 1
    tmax = 500
    epv = [0, 0.01, 0.1, 1]
    deltav = [0.25, 0.5, 0.75, 0.9, 0.95]

    data = np.zeros((6, len(epv), len(deltav)))

    for ep in epv:
        for delta in deltav:

            rwd_bas, rwd_bas_dev, rwd_lew = simulate(tmax, actions, a1m, a2m, delta, ep)

            crw_bas_1 = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_bas[0, :])
            crw_bas_2 = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_bas[1, :])
            crw_bas_dev_1 = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_bas_dev[0, :])
            crw_bas_dev_2 = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_bas_dev[1, :])
            crw_lew_1 = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_lew[0, :])
            crw_lew_2 = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_lew[1, :])
            data[:, epv.index(ep), deltav.index(delta)] = np.array([crw_bas_1, crw_bas_2, crw_lew_1, crw_lew_2, crw_bas_dev_1, crw_bas_dev_2])

            if epv.index(ep) > len(epv) - 2:  # Plot only once (ep=0.1)
                plt.plot(np.array(pv)[:, 0], np.array(pv)[:, 1], 'ob')
                plt.plot(p1n, p2n, 'or')
                plt.plot(pi1[-1], pi2[-1], 'ok')
                plt.plot(crw_bas_1, crw_bas_2, 'gx')
                plt.plot(crw_bas_dev_1, crw_bas_dev_2, 'cx')
                plt.plot(crw_lew_1, crw_lew_2, 'ks')
                plt.title('Payoffs for delta ' + str(delta))
                save('base_'+str(delta)+'.tex')
                plt.show()
                print('Delta value = ', delta)
                print('Baseline payoff ', crw_bas_1, crw_bas_2)
                print('Nash payoff ', p1n, p2n)
                print('Baseline with dev ', crw_bas_dev_1, crw_bas_dev_2)
                print('LEWIS ', crw_lew_1, crw_lew_2)

    for i, delta in enumerate(deltav):
        plt.plot(epv, data[0, :, i], 'ro-', label='baseline')
        plt.plot(epv, data[1, :, i], 'rx-')
        plt.plot(epv, data[2, :, i], 'bo-', label='LEWIS')
        plt.plot(epv, data[3, :, i], 'bx-')
        plt.plot(epv, data[4, :, i], 'ko-', label='Baseline with dev')
        plt.plot(epv, data[5, :, i], 'kx-')
        plt.legend(loc='best')
        plt.xlabel('Epsilon')
        plt.ylabel('Cum reward')
        plt.title('Results for delta ' + str(delta))
        plt.show()

    print('Done')