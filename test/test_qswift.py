import math

from qwrapper.obs import Pauli
from scipy.linalg import expm
import numpy as np
from unittest import TestCase


def build_probs(params, lam):
    probs = [p / lam for p in params]
    probs.append(1 - sum(probs))
    probs = np.array(probs, dtype=np.complex128)
    return probs


def apply_l(rho, operator):
    return 1j * (operator.dot(rho) - rho.dot(operator))


def apply(rho, probs, operators, lam, N):
    current = None
    for prob, operator in zip(probs, operators):
        value = prob * expm(1j * operator * lam / N).dot(rho).dot(expm(-1j * operator * lam / N))
        if current is None:
            current = value
        else:
            current += value
    return current


def apply_q1_l(rho, probs, operators):
    return apply_qn_l0(rho, probs, operators, 1)


def apply_qn_l0(rho, probs, operators, n):
    current = rho
    for _ in range(n):
        tmp = None
        for p, o in zip(probs, operators):
            value = p * apply_l(current, o)
            if tmp is None:
                tmp = value
            else:
                tmp = tmp + value
        current = tmp
    return current


def apply_qn_l(rho, probs, operators, n):
    current0 = apply_qn_l0(rho, probs, operators, n)
    current1 = None
    for p, o in zip(probs, operators):
        tmp = rho
        for _ in range(n):
            tmp = apply_l(tmp, o)
        if current1 is None:
            current1 = p * tmp
        else:
            current1 = current1 + p * tmp
    return current0 - current1


def fo_value(obs, operators, params, lam, rho, N):
    probs = build_probs(params, lam)
    for j in range(N):
        rho = apply(rho, probs, operators, lam, N)
    return np.trace(obs.dot(rho)).real


def so_correction(obs, operators, params, lam, rho, N):
    probs = build_probs(params, lam)
    coeff = lam ** 2 / (2 * N)
    result = None
    for ell in range(N):
        current = rho
        for index in range(N):
            if index == ell:
                current = apply_qn_l(current, probs, operators, 2)
            else:
                current = apply(current, probs, operators, lam, N)
        if result is None:
            result = current / N
        else:
            result = result + current / N
    return coeff * np.trace(obs.dot(result)).real


def so_value(obs, operators, params, lam, rho, N):
    return fo_value(obs, operators, params, lam, rho, N) \
           + so_correction(obs, operators, params, lam, rho, N)


def diff_grad(obs, operators, params, lam, rho, N, k, delta, func):
    ps = []
    for index, p in enumerate(params):
        if index == k:
            ps.append(p + delta)
        else:
            ps.append(p)
    grad = (func(obs, operators, ps, lam, rho, N) - func(obs, operators, params, lam, rho, N)) / delta
    return grad


def fo_diff_grad(obs, operators, params, lam, rho, N, k, delta):
    return diff_grad(obs, operators, params, lam, rho, N, k, delta, fo_value)


def so_diff_grad(obs, operators, params, lam, rho, N, k, delta):
    return diff_grad(obs, operators, params, lam, rho, N, k, delta, so_value)


def expansion(rho, operators, lam, N, k):
    result = None
    for n in range(1, 10):
        tmp = rho
        coeff = 1 / math.factorial(n) * (lam / N) ** (n - 1)
        for _ in range(n):
            tmp = apply_l(tmp, operators[k])
        if result is None:
            result = coeff * tmp
        else:
            result = result + coeff * tmp
    return result


def fo_analytic_grad(obs, operators, params, rho, lam, N, k):
    probs = build_probs(params, lam)
    result = None
    for ell in range(N):
        current = rho
        for index in range(N):
            if index == ell:
                current = expansion(current, operators, lam, N, k)
            else:
                current = apply(current, probs, operators, lam, N)
        if result is None:
            result = current
        else:
            result = result + current
    return np.trace(obs.dot(result)).real / N


def so_analytic_grad(obs, operators, params, rho, lam, N, k):
    return fo_analytic_grad(obs, operators, params, rho, lam, N, k) + \
           so_correction_analytic_grad(obs, operators, params, rho, lam, N, k)


def so_correction_analytic_grad(obs, operators, params, rho, lam, N, k):
    probs = build_probs(params, lam)
    coeff = lam ** 2 / (2 * N)
    one = d1(operators, probs, rho, lam, N, k)
    two = d2(operators, probs, rho, lam, N, k)
    one = coeff * np.trace(obs.dot(one)).real
    two = coeff * np.trace(obs.dot(two)).real
    result = one + two
    return result


def d1(operators, probs, rho, lam, N, k):
    coeff = 1 / lam
    tmp = None
    for ell in range(N):
        current = rho
        for index in range(N):
            if index == ell:
                c1 = apply_l(apply_q1_l(current, probs, operators), operators[k])
                c2 = apply_q1_l(apply_l(current, operators[k]), probs, operators)
                c3 = -apply_l(apply_l(current, operators[k]), operators[k])
                current = c1 + c2 + c3
            else:
                current = apply(current, probs, operators, lam, N)
        if tmp is None:
            tmp = current / N
        else:
            tmp = tmp + current / N
    return coeff * tmp


def d2(operators, probs, rho, lam, N, k):
    result = None
    for ell1 in range(N):
        for ell2 in range(N):
            current = rho
            if ell2 == ell1:
                continue
            for index in range(N):
                if index == ell1:
                    current = apply_qn_l(current, probs, operators, 2)
                elif index == ell2:
                    current = expansion(current, operators, lam, N, k)
                else:
                    current = apply(current, probs, operators, lam, N)
            if result is None:
                result = current
            else:
                result = result + current
    return result / N ** 2


def exact_value(obs, operators, params, rho):
    result = None
    for param, operator in zip(params, operators[:len(operators) - 1]):
        if result is None:
            result = param * operator
        else:
            result = result + param * operator
    return np.trace(obs.dot(expm(1j * result).dot(rho).dot(expm(-1j * result)))).real


class TestGradientTheory(TestCase):
    def test_theory(self):
        obs = Pauli.X
        operators = [Pauli.Z, Pauli.Y, Pauli.I]
        params = np.array([0.1, 0.2])
        lam = 2
        rho = np.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]], dtype=np.complex128)
        N = 4
        print("energy", fo_value(obs, operators, params, lam, rho, N),
              so_value(obs, operators, params, lam, rho, N),
              exact_value(obs, operators, params, rho))
        print("------first-order------")
        for j in range(len(params)):
            ans = fo_diff_grad(obs, operators, params, lam, rho, N, j, 0.0001)
            result = fo_analytic_grad(obs, operators, params, rho, lam, N, j)
            self.assertAlmostEqual(ans, result, delta=0.001)
            print(f"grad:{j}", ans, result)

        print("------second-order------")
        for j in range(len(params)):
            ans = so_diff_grad(obs, operators, params, lam, rho, N, j, 0.0001)
            result = so_analytic_grad(obs, operators, params, rho, lam, N, j)
            self.assertAlmostEquals(ans, result, delta=0.001)
            print(f"grad:{j}", ans, result)
