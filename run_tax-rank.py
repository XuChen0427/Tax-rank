import matplotlib
import cvxpy as cp
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import ot
import argparse
from mpl_toolkits.mplot3d import Axes3D


def accuracy(UI_matrix,gamma, x):
    w = np.zeros(UI_matrix.shape)

    for i in range(len(UI_matrix)):
        w[i,:] = UI_matrix[i,:] * gamma

    user_size = x.shape[0]
    acc = np.sum(w*x) / user_size
    return acc

def gini(UI_matrix, gamma, x):

    w = np.zeros(UI_matrix.shape)

    for i in range(len(UI_matrix)):
        w[i,:] = UI_matrix[i,:] * gamma
    exposure = np.sum(w * x, axis=0)
    exposure = sorted(exposure, reverse=False)
    total = 0
    for i, xi in enumerate(exposure[:-1], 1):
        total += np.sum(np.abs(xi-exposure[i:]))
    return total / (len(exposure)**2 * np.mean(exposure))




def solve_OT(e, W, args):
    # W shape = a * b
    # a = user size
    # b = item size
    Ks = [args.topk for i in range(args.U)]
    answer = ot.sinkhorn(Ks, e, -W, args.lbd)
    # print(f'answer')
    # answer = ot.optim.cg(Ks, e, -W, lbd, f, df, verbose=True)

    answer = np.where(np.isnan(answer), 0, answer)
    answer = np.clip(answer, 0, 1)
    recommend_list = answer

    #
    # print(sum(recommend_list[0, :]))
    # recommend_list = np.zeros(answer.shape)
    # indexs = answer.argsort()[:, ::-1][:, :K]
    # for i in range(len(indexs)):
    #     recommend_list[i, indexs[i]] = 1

    return recommend_list



def lower_bound(UI_matrix, gamma, args):

    user_size, item_size = UI_matrix.shape

    e = cp.Variable(item_size)

    con = [e >= 0, e <= 1, cp.sum(e) == args.topk]

    eta = args.k * np.sum(UI_matrix, axis=0).reshape(item_size)

    if args.t == 1:
        fairness = cp.sum(gamma * cp.multiply(eta, cp.log(e)))
    else:

        fairness = cp.sum(gamma * cp.multiply(eta, cp.power(e, 1-args.t) / (1 - args.t)))

    obj = cp.Maximize(fairness)

    prob = cp.Problem(obj, con)
    prob.solve(solver='MOSEK', verbose=True)
    print(f'__________t{args.t} obj:', prob.value)
    # print('最优解：', x.value)
    return e.value


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-rank")
    parser.add_argument('--U', type=int, default='503')
    parser.add_argument('--I', type=int, default='314')
    parser.add_argument('--mode', type=str, default='rec')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--t', type=float, default=0.0)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--lbd', type=float, default=1)
    args = parser.parse_args()
    if args.mode == 'rec':
        UI_matrix = np.load('./simulator/yelp/matrix.npy')
        UI_matrix = sigmoid(UI_matrix)
        user_size, item_size = UI_matrix.shape
        item_weight = np.ones(item_size)
    else:
        UI_matrix = np.load('./simulator/ipinyou/ipinyou_small.npy')
        UI_matrix = sigmoid(UI_matrix)
        user_size, item_size = UI_matrix.shape
        item_weight = np.load('./simulator/ipinyou/ipinyou_weight.npy')
        item_weight = np.log10(item_weight)

    print(f'user_size:{user_size}, item_size:{item_size}')

    UI_matrix = UI_matrix[:args.U, :args.I]
    item_weight = item_weight[:args.I]
    user_size, item_size = UI_matrix.shape
    print(f'after user_size:{user_size}, item_size:{item_size}')


    gamma = item_weight
    e = lower_bound(UI_matrix, gamma=item_weight, args=args)
    print(f' e:{e}')
    x = solve_OT(e, UI_matrix, args)


    w = UI_matrix
    gini = gini(w, gamma,x)
    accuracy = accuracy(w,gamma, x)

    print(f'GINI:{gini} ACC:{accuracy}')




