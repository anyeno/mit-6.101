#!/usr/bin/env python3
import typing
import doctest


# NO ADDITIONAL IMPORTS ALLOWED!
import os
import sys
import pickle
import doctest


import lab

sys.setrecursionlimit(20000)

TEST_DIRECTORY = os.path.dirname(__file__)

TESTDOC_FLAGS = doctest.NORMALIZE_WHITESPACE | doctest.REPORT_ONLY_FIRST_FAILURE
TESTDOC_SKIP = ['lab']

def dump(game):
    """
    Prints a human-readable version of a game (provided as a dictionary)
    """
    for key, val in sorted(game.items()):
        if isinstance(val, list) and val and isinstance(val[0], list):
            print(f"{key}:")
            for inner in val:
                print(f"    {inner}")
        else:
            print(f"{key}:", val)


# 2-D IMPLEMENTATION


def new_game_2d(num_rows, num_cols, bombs):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'hidden' fields adequately initialized.

    Parameters:
       num_rows (int): Number of rows
       num_cols (int): Number of columns
       bombs (list): List of bombs, given in (row, column) pairs, which are
                     tuples

    Returns:
       A game state dictionary

    >>> dump(new_game_2d(2, 4, [(0, 0), (1, 0), (1, 1)]))
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: (2, 4)
    hidden:
        [True, True, True, True]
        [True, True, True, True]
    state: ongoing
    """
    board = []
    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            if [r, c] in bombs or (r, c) in bombs:
                row.append(".")
            else:
                row.append(0)
        board.append(row)
    hidden = []
    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            row.append(True)
        hidden.append(row)
    for r in range(num_rows):
        for c in range(num_cols):
            if board[r][c] == 0:
                board[r][c] = find_neighbor_bombs(r, c, board)
    return {
        "dimensions": (num_rows, num_cols),
        "board": board,
        "hidden": hidden,
        "state": "ongoing",
    }


def find_neighbor_bombs(r, c, board):
    res = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            dr = r + i
            dc = c + j
            if 0 <= dr < len(board) and 0 <= dc < len(board[0]):
                if board[dr][dc] == '.':
                    res += 1
    return res


def dig_2d(game, row, col):
    """
    Reveal the cell at (row, col), and, in some cases, recursively reveal its
    neighboring squares.

    Update game['hidden'] to reveal (row, col).  Then, if (row, col) has no
    adjacent bombs (including diagonally), then recursively reveal (dig up) its
    eight neighbors.  Return an integer indicating how many new squares were
    revealed in total, including neighbors, and neighbors of neighbors, and so
    on.

    The state of the game should be changed to 'defeat' when at least one bomb
    is revealed on the board after digging (i.e. game['hidden'][bomb_location]
    == False), 'victory' when all safe squares (squares that do not contain a
    bomb) and no bombs are revealed, and 'ongoing' otherwise.

    Parameters:
       game (dict): Game state
       row (int): Where to start digging (row)
       col (int): Where to start digging (col)

    Returns:
       int: the number of new squares revealed

    >>> game = {'dimensions': (2, 4),
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'hidden': [[True, False, True, True],
    ...                  [True, True, True, True]],
    ...         'state': 'ongoing'}
    >>> dig_2d(game, 0, 3)
    4
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: (2, 4)
    hidden:
        [True, False, False, False]
        [True, True, False, False]
    state: victory

    >>> game = {'dimensions': [2, 4],
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'hidden': [[True, False, True, True],
    ...                  [True, True, True, True]],
    ...         'state': 'ongoing'}
    >>> dig_2d(game, 0, 0)
    1
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: [2, 4]
    hidden:
        [False, False, True, True]
        [True, True, True, True]
    state: defeat
    """
    if game["state"] == "defeat" or game["state"] == "victory":
        game["state"] = game["state"]  # keep the state the same
        return 0

    if game["board"][row][col] == ".":
        game["hidden"][row][col] = False
        game["state"] = "defeat"
        return 1

    bombs = 0  # 显示出来的地雷数量
    hidden_squares = 0  # hidden的非地雷的格子数量
    for r in range(game["dimensions"][0]):
        for c in range(game["dimensions"][1]):
            if game["board"][r][c] == ".":
                if game["hidden"][r][c] == False:
                    bombs += 1
            elif game["hidden"][r][c] == True:
                hidden_squares += 1
    if bombs != 0:
        # if bombs is not equal to zero, set the game state to defeat and
        # return 0
        game["state"] = "defeat"
        return 0
    if hidden_squares == 0:
        game["state"] = "victory"
        return 0

    if game["hidden"][row][col] != False:
        game["hidden"][row][col] = False
        revealed = 1
    else:
        return 0

    if game["board"][row][col] == 0:
        num_rows, num_cols = game["dimensions"]
        for i in range(-1, 2):
            for j in range(-1, 2):
                dr = row + i
                dc = col + j
                if 0 <= dr < num_rows and 0 <= dc < num_cols:
                    if game['hidden'][dr][dc]:
                        revealed += dig_2d(game, dr, dc)

    bombs = 0  # set number of bombs to 0
    hidden_squares = 0
    for r in range(game["dimensions"][0]):
        # for each r,
        for c in range(game["dimensions"][1]):
            # for each c,
            if game["board"][r][c] == ".":
                if game["hidden"][r][c] == False:
                    # if the game hidden is False, and the board is '.', add 1 to
                    # bombs
                    bombs += 1
            elif game["hidden"][r][c] == True:
                hidden_squares += 1
    bad_squares = bombs + hidden_squares
    if bad_squares > 0:
        game["state"] = "ongoing"
        return revealed
    else:
        game["state"] = "victory"
        return revealed


def render_2d_locations(game, xray=False):
    """
    Prepare a game for display.

    Returns a two-dimensional array (list of lists) of '_' (hidden squares),
    '.' (bombs), ' ' (empty squares), or '1', '2', etc. (squares neighboring
    bombs).  game['hidden'] indicates which squares should be hidden.  If
    xray is True (the default is False), game['hidden'] is ignored and all
    cells are shown.

    Parameters:
       game (dict): Game state
       xray (bool): Whether to reveal all tiles or just the that are not
                    game['hidden']

    Returns:
       A 2D array (list of lists)

    >>> render_2d_locations({'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'hidden':  [[True, False, False, True],
    ...                   [True, True, False, True]]}, False)
    [['_', '3', '1', '_'], ['_', '_', '1', '_']]

    >>> render_2d_locations({'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'hidden':  [[True, False, True, False],
    ...                   [True, True, True, False]]}, True)
    [['.', '3', '1', ' '], ['.', '.', '1', ' ']]
    """
    res = []
    board = game['board']
    hidden = game['hidden']
    if xray:
        for b in board:
            t = []
            for i in b:
                t.append(str(i)) if i != 0 else t.append(' ')
            res.append(t)
        return res

    for i in range(len(hidden)):
        t = []
        for j in range(len(hidden[0])):
            if not hidden[i][j]:
                t.append(str(board[i][j])) if board[i][j] != 0 else t.append(' ')
            else:
                t.append('_')
        res.append(t)

    return res


def render_2d_board(game, xray=False):
    """
    Render a game as ASCII art.

    Returns a string-based representation of argument 'game'.  Each tile of the
    game board should be rendered as in the function
        render_2d_locations(game)

    Parameters:
       game (dict): Game state
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    game['hidden']

    Returns:
       A string-based representation of game

    >>> render_2d_board({'dimensions': (2, 4),
    ...                  'state': 'ongoing',
    ...                  'board': [['.', 3, 1, 0],
    ...                            ['.', '.', 1, 0]],
    ...                  'hidden':  [[False, False, False, True],
    ...                            [True, True, False, True]]})
    '.31_\\n__1_'
    """
    locations = render_2d_locations(game, xray)
    res = ""
    for i in range(len(locations)):
        for j in range(len(locations[0])):
            res += locations[i][j]
        if i != len(locations) - 1:
            res += '\n'

    return res


# N-D IMPLEMENTATION

# 创建给定维度列表的数组，初始化为value
def create_nd_array(dimensions, value):
    """
    >>> create_nd_array([3,3,3], 1)
    [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    """
    if len(dimensions) == 1:
        res = [value] * dimensions[0]
        return res
    res = []
    for i in range(dimensions[0]):
        res.append(create_nd_array(dimensions[1:], value))
    return res


# 给定一个 Nd 数组和一个元组/坐标列表，返回数组中这些坐标处的值。
def find_nd_value(nd_array, to_find):
    if len(to_find) == 1:
        return nd_array[to_find[0]]
    else:
        return find_nd_value(nd_array[to_find[0]], to_find[1:])


# 给定一个 Nd 数组、一个元组/坐标列表和一个值，用给定值替换数组中这些坐标处的值。
def change_nd_value(nd_array, to_change, value):
    if len(to_change) == 1:
        nd_array[to_change[0]] = value
    else:
        change_nd_value(nd_array[to_change[0]], to_change[1:], value)


# 返回给定坐标的所有邻居(包括自己)
def find_neighbors(dimensions, location):
    res = []
    for i in range(-1, 2):
        t = location[0] + i
        if 0 <= t < dimensions[0]:
            res.append(t)
    if len(location) == 1:
        return res
    result = []
    for i in find_neighbors(dimensions[1:], location[1:]):
        for j in res:
            if type(i) is int:
                result.append([j] + [i])
            else:
                result.append([j] + i)
    return result


# 返回给定棋盘中所有可能坐标的函数
def all_locations(dimensions):
    res = []
    t = []
    for i in range(dimensions[0]):
        t.append(i)
    if len(dimensions) == 1:
        return t
    for i in all_locations(dimensions[1:]):
        for j in t:
            if type(i) is int:
                res.append([j] + [i])
            else:
                res.append([j] + i)
    return res


def new_game_nd(dimensions, bombs):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'hidden' fields adequately initialized.


    Args:
       dimensions (tuple): Dimensions of the board
       bombs (list): Bomb locations as a list of tuples, each an
                     N-dimensional coordinate

    Returns:
       A game state dictionary

    >>> g = new_game_nd((2, 4, 2), [(0, 0, 1), (1, 0, 0), (1, 1, 1)])
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    hidden:
        [[True, True], [True, True], [True, True], [True, True]]
        [[True, True], [True, True], [True, True], [True, True]]
    state: ongoing
    """
    board = create_nd_array(dimensions, 0)
    for bomb in bombs:
        change_nd_value(board, bomb, '.')
    hidden = create_nd_array(dimensions, True)

    # 更新炸弹值 board
    all_loc = all_locations(dimensions)
    for loc in all_loc:
        if find_nd_value(board, loc) == 0:
            bombs = 0
            neighbors = find_neighbors(dimensions, loc)  # 该点的所有邻居
            neighbors.remove(loc)  # 把自己从邻居里删去
            for nei in neighbors:
                if find_nd_value(board, nei) == ".":  # 如果这个邻居是炸弹
                    bombs += 1
            change_nd_value(board, loc, bombs)

    return {
        "dimensions": dimensions,
        "board": board,
        "hidden": hidden,
        "state": "ongoing",
    }


def dig_nd(game, coordinates):
    """
    Recursively dig up square at coords and neighboring squares.

    Update the hidden to reveal square at coords; then recursively reveal its
    neighbors, as long as coords does not contain and is not adjacent to a
    bomb.  Return a number indicating how many squares were revealed.  No
    action should be taken and 0 returned if the incoming state of the game
    is not 'ongoing'.

    The updated state is 'defeat' when at least one bomb is revealed on the
    board after digging, 'victory' when all safe squares (squares that do
    not contain a bomb) and no bombs are revealed, and 'ongoing' otherwise.

    Args:
       coordinates (tuple): Where to start digging

    Returns:
       int: number of squares revealed

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'hidden': [[[True, True], [True, False], [True, True],
    ...                [True, True]],
    ...               [[True, True], [True, True], [True, True],
    ...                [True, True]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 3, 0))
    8
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    hidden:
        [[True, True], [True, False], [False, False], [False, False]]
        [[True, True], [True, True], [False, False], [False, False]]
    state: ongoing
    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'hidden': [[[True, True], [True, False], [True, True],
    ...                [True, True]],
    ...               [[True, True], [True, True], [True, True],
    ...                [True, True]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 0, 1))
    1
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    hidden:
        [[True, False], [True, False], [True, True], [True, True]]
        [[True, True], [True, True], [True, True], [True, True]]
    state: defeat
    """
    coordinates = list(coordinates)
    # print(coordinates)
    if game["state"] == "defeat" or game["state"] == "victory":
        game["state"] = game["state"]  # keep the state the same
        return 0
    board = game['board']
    hidden = game['hidden']
    dimensions = game['dimensions']
    if find_nd_value(board, coordinates) == ".":
        change_nd_value(hidden, coordinates, False)
        game["state"] = "defeat"
        return 1

    bombs = 0  # 显示出来的地雷数量
    hidden_squares = 0  # hidden的非地雷的格子数量
    all_loc = all_locations(dimensions)  # 所有的坐标
    for loc in all_loc:
        hidden_value = find_nd_value(hidden, loc)
        if find_nd_value(board, loc) == ".":
            if hidden_value == False:
                bombs += 1
        elif hidden_value == True:
            hidden_squares += 1
    if bombs != 0:
        # if bombs is not equal to zero, set the game state to defeat and
        # return 0
        game["state"] = "defeat"
        return 0
    if hidden_squares == 0:
        game["state"] = "victory"
        return 0

    if find_nd_value(hidden, coordinates) != False:
        change_nd_value(hidden, coordinates, False)
        revealed = 1
    else:
        return 0

    if find_nd_value(board, coordinates) == 0:
        # print("board == 0")
        neighbors = find_neighbors(dimensions, coordinates)
        neighbors.remove(coordinates)   # 把自己从邻居中删除
        # print("neighbors == ")
        # print(neighbors)
        for nei in neighbors:
            if find_nd_value(hidden, nei) == True:
                revealed += dig_nd(game, nei)

    bombs = 0  # set number of bombs to 0
    hidden_squares = 0
    for loc in all_loc:
        hidden_value = find_nd_value(hidden, loc)
        if find_nd_value(board, loc) == ".":
            if hidden_value == False:
                # if the game hidden is False, and the board is '.', add 1 to
                # bombs
                bombs += 1
        elif hidden_value == True:
            hidden_squares += 1
    bad_squares = bombs + hidden_squares
    if bad_squares > 0:
        game["state"] = "ongoing"
        return revealed
    else:
        game["state"] = "victory"
        return revealed


def render_nd(game, xray=False):
    """
    Prepare the game for display.

    Returns an N-dimensional array (nested lists) of '_' (hidden squares), '.'
    (bombs), ' ' (empty squares), or '1', '2', etc. (squares neighboring
    bombs).  The game['hidden'] array indicates which squares should be
    hidden.  If xray is True (the default is False), the game['hidden'] array
    is ignored and all cells are shown.

    Args:
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    game['hidden']

    Returns:
       An n-dimensional array of strings (nested lists)

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'hidden': [[[True, True], [True, False], [False, False],
    ...                [False, False]],
    ...               [[True, True], [True, True], [False, False],
    ...                [False, False]]],
    ...      'state': 'ongoing'}
    >>> render_nd(g, False)
    [[['_', '_'], ['_', '3'], ['1', '1'], [' ', ' ']],
     [['_', '_'], ['_', '_'], ['1', '1'], [' ', ' ']]]

    >>> render_nd(g, True)
    [[['3', '.'], ['3', '3'], ['1', '1'], [' ', ' ']],
     [['.', '3'], ['3', '.'], ['1', '1'], [' ', ' ']]]
    """
    dimensions = game['dimensions']
    board = game['board']
    hidden = game['hidden']
    res = create_nd_array(dimensions, ' ')
    all_loc = all_locations(dimensions)

    if xray:
        for loc in all_loc:
            v = find_nd_value(board, loc)
            if v != 0:
                change_nd_value(res, loc, str(v))
        return res
    else:
        for loc in all_loc:
            if find_nd_value(hidden, loc) == False:
                v = find_nd_value(board, loc)
                if v != 0:
                    change_nd_value(res, loc, str(v))
            else:
                change_nd_value(res, loc, '_')

        return res


# def test_nd_integration(test):
#     exp_fname = os.path.join(TEST_DIRECTORY, 'test_outputs', f'testnd_integration{test}.pickle')
#     inp_fname = os.path.join(TEST_DIRECTORY, 'test_inputs', f'testnd_integration{test}.pickle')
#     with open(exp_fname, 'rb') as f:
#         expected = pickle.load(f)
#     with open(inp_fname, 'rb') as f:
#         inputs = pickle.load(f)
#     g = lab.new_game_nd(inputs['dimensions'], inputs['bombs'])
#     print(g['dimensions'])
#     # print(g['board'])
#     for location, results in zip(inputs['digs'], expected):
#         squares_revealed, game, rendered, rendered_xray = results
#         res = lab.dig_nd(g, location)
#         assert res == squares_revealed
#         for i in ('dimensions', 'board', 'hidden', 'state'):
#             assert g[i] == game[i]
#         assert lab.render_nd(g) == rendered
#         assert lab.render_nd(g, True) == rendered_xray

if __name__ == "__main__":
    # Test with doctests. Helpful to debug individual lab.py functions.
    _doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    doctest.testmod(optionflags=_doctest_flags)  # runs ALL doctests

    # Alternatively, can run the doctests JUST for specified function/methods,
    # e.g., for render_2d_locations or any other function you might want.  To
    # do so, comment out the above line, and uncomment the below line of code.
    # This may be useful as you write/debug individual doctests or functions.
    # Also, the verbose flag can be set to True to see all test results,
    # including those that pass.
    #
    # doctest.run_docstring_examples(
    #    render_2d_locations,
    #    globals(),
    #    optionflags=_doctest_flags,
    #    verbose=False
    # )

    # dim3 = [3,3,3]
    # dim2 = [2,3]
    # # all_loc = all_locations(dim3)
    # nei = find_neighbors(dim3, [0,0,0])
    # print(len(nei))
    # print(nei)
    # test_nd_integration(1)
    pass
