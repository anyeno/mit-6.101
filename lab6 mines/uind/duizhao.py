def dig_2d(game, row, col):
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