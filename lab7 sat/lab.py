#!/usr/bin/env python3
import copy
import sys
import typing
import doctest

sys.setrecursionlimit(10_000)


# NO ADDITIONAL IMPORTS

def update(formula, variable, bool_value):
    """
    >>> x = update([[('a', True), ('b', True), ('c', False)], [('c', True), ('d', True)]], 'c', True)
    >>> x
    [[('a', True), ('b', True)]]
    >>> y = update([[('c', False)], [('c', True), ('d', True)]], 'c', True)
    >>> y
    False
    >>> z = update([[], [], []], 'a', True)
    >>> z
    []
    """
    res = []
    for clause in formula:
        if len(clause) == 0:
            continue
        new_clause = []
        flag = 1
        for var in clause:
            if variable == var[0]:
                if bool_value:
                    if var[1]:
                        flag = 0
                        break
                    else:
                        if len(clause) == 1:
                            return False
                else:
                    if not var[1]:
                        flag = 0
                        break
                    else:
                        if len(clause) == 1:
                            return False
            else:
                new_clause.append(var)
        if flag == 1 and len(new_clause) > 0:
            res.append(new_clause)

    return res


def helper_satisfying_ass(form):
    # print("form == ")
    # print(form)
    formula = copy.deepcopy(form)
    if len(formula) == 0:
        return {}
    res = {}
    for clause in formula:
        assert len(clause) > 0
        if len(clause) == 1:
            res[clause[0][0]] = clause[0][1]
    for k, v in res.items():
        formula = update(formula, k, v)
        if formula is False:
            return False

    if len(formula) == 0:
        return res

    assert len(formula[0]) > 0
    child1 = update(formula, formula[0][0][0], True)
    child2 = update(formula, formula[0][0][0], False)
    if child1 is False and child2 is False:
        return False
    if child1 is not False:
        child_res = helper_satisfying_ass(child1)
        if child_res is not False:
            child_res[formula[0][0][0]] = True
            result = {**res, **child_res}
            return result
    if child2 is not False:
        child_res = helper_satisfying_ass(child2)
        if child_res is False:
            return False
        child_res[formula[0][0][0]] = False
        result = {**res, **child_res}
        return result
    return False


def satisfying_assignment(formula):
    """
    Find a satisfying assignment for a given CNF formula.
    Returns that assignment if one exists, or None otherwise.

    >>> satisfying_assignment([])
    {}
    >>> x = satisfying_assignment([[('a', True), ('b', False), ('c', True)]])
    >>> x.get('a', None) is True or x.get('b', None) is False or x.get('c', None) is True
    True
    >>> satisfying_assignment([[('a', True)], [('a', False)]])
    """
    new_formula = []
    for clause in formula:
        if len(clause) > 0:
            new_formula.append(clause)
    res = helper_satisfying_ass(new_formula)
    if res is False:
        return None

    return res


def sudoku_board_to_sat_formula(sudoku_board):
    """
    Generates a SAT formula that, when solved, represents a solution to the
    given sudoku board.  The result should be a formula of the right form to be
    passed to the satisfying_assignment function above.
    """
    n = len(sudoku_board)
    res = []
    # 每个单元格有一个数字
    for i in range(n):
        for j in range(n):
            clause = []
            for k in range(n):
                clause.append(([k + 1, i, j], True))        # k in (i, j)
            res.append(clause)
    # 每个单元只有一个数字
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for p in range(k+1, n):
                    clause = [([k + 1, i, j], False), ([p + 1, i, j], False)]
                    res.append(clause)
    # 与题面sudoku_board一致
    for i in range(n):
        for j in range(n):
            number = sudoku_board[i][j]
            if number != 0:
                clause = [([number, i, j], True)]
                res.append(clause)
    # 每一行都是1-n
    for k in range(n):
        for i in range(n):
            for j1 in range(n):
                for j2 in range(j1 + 1, n):
                    clause = [([k + 1, i, j1], False), ([k + 1, i, j2], False)]
                    res.append(clause)
    # 每一列都是1-n
    for k in range(n):
        for j in range(n):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    clause = [([k + 1, i1, j], False), ([k + 1, i2, j], False)]
                    res.append(clause)
    # 每个根号n的方格里是1-n
    n_sqrt = int(n ** 0.5)
    num = n // n_sqrt
    # print("num === " + str(num))
    # print("n_sqrt === " + str(n_sqrt))
    for no in range(num):
        for k in range(n):
            for i1 in range(no * n_sqrt, (no + 1) * n_sqrt):
                for i2 in range(no * n_sqrt, (no + 1) * n_sqrt):
                    for inner_no in range(num):
                        for j1 in range(inner_no * n_sqrt, (inner_no + 1) * n_sqrt):
                            for j2 in range(inner_no * n_sqrt, (inner_no + 1) * n_sqrt):
                                if i1 == i2 and j1 == j2:
                                    continue
                                else:
                                    # print(str(i1) + " " + str(j1) + "   " + str(i2) + " " + str(j2))
                                    clause = [([k + 1, i1, j1], False), ([k + 1, i2, j2], False)]
                                    res.append(clause)
    # print(res)
    string_res = []
    for clause in res:
        new_cla = []
        for variable in clause:
            v = variable[0]
            list2string = str(v[0]) + '#' + str(v[1]) + '#' + str(v[2])
            new_cla.append((list2string, variable[1]))
        string_res.append(new_cla)
    # print(string_res)
    return string_res


def assignments_to_sudoku_board(assignments, n):
    """
    Given a variable assignment as given by satisfying_assignment, as well as a
    size n, construct an n-by-n 2-d array (list-of-lists) representing the
    solution given by the provided assignment of variables.

    If the given assignments correspond to an unsolveable board, return None
    instead.
    """
    if assignments is None:
        return None
    res = [[0] * n for i in range(n)]
    # count = 0
    for k, v in assignments.items():
        if v:
            # count += 1
            location = k.split("#")
            # print(location)
            i = int(location[1])
            j = int(location[2])
            number = int(location[0])
            # print("i == " + str(i))
            # print("j == " + str(j))
            # print("number == ")
            # print(number)
            # print(res)
            res[i][j] = number
            # print("res =============== ")
            # print(res)
    # print("count ===  " + str(count))
    return res


if __name__ == "__main__":
    import doctest
    #
    # _doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    # doctest.testmod(optionflags=_doctest_flags)
    grid = [
        [0, 0, 0, 2],
        [0, 0, 0, 1],
        [4, 0, 0, 0],
        [2, 0, 0, 0],
    ]
    # grid[0][0] = 9999999
    # print(grid)
    # formual = sudoku_board_to_sat_formula(grid)
    # ass = satisfying_assignment(formual)
    # print(ass)
    # res = assignments_to_sudoku_board(ass, 4)
    # print(res)
    grid5 = [
        [5, 0, 1, 8, 0, 3, 7, 0, 2],  # http://www.extremesudoku.info/sudoku.html
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [7, 0, 0, 2, 0, 5, 0, 0, 8],
        [6, 0, 2, 0, 0, 0, 4, 0, 7],
        [0, 0, 0, 0, 5, 6, 0, 0, 0],
        [1, 0, 7, 0, 0, 0, 9, 0, 5],
        [8, 0, 0, 9, 0, 2, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 9, 5, 0, 7, 6, 0, 1],
    ]
    f = sudoku_board_to_sat_formula(grid5)
    ass = satisfying_assignment(f)
    res = assignments_to_sudoku_board(ass, 9)
    for clause in res:
        print(clause)
    # print(res)

