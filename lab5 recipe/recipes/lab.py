# Recipes Database
# NO ADDITIONAL IMPORTS!
import copy
import sys

sys.setrecursionlimit(20_000)


def replace_item(recipes, old_name, new_name):
    """
    Returns a new recipes list based on the input list, where all mentions of
    the food item given by old_name are replaced with new_name.
    """
    res = []
    for recipe in recipes:
        type = recipe[0]
        name = recipe[1]
        if name == old_name:
            name = new_name
        if type == 'compound':
            cost = []
            for i in recipe[2]:
                if i[0] == old_name:
                    cost.append((new_name, i[1]))
                else:
                    cost.append(i)
        else:
            cost = recipe[2]
        res.append((type, name, cost))

    return res


def helper_lowest_cost(recipes, food_item):
    if food_item == "null":
        return None
    min_cost = 1e8
    flag = False
    for recipe in recipes:
        if recipe[1] == food_item:
            flag = True
            if recipe[0] == 'atomic':
                return recipe[2]
            else:
                temp = 0
                child_is_none = False
                for i in recipe[2]:
                    lowest_cost_child = helper_lowest_cost(recipes, i[0])
                    if lowest_cost_child is None:
                        child_is_none = True
                        break
                    temp += lowest_cost_child * i[1]
                if not child_is_none:
                    min_cost = min(min_cost, temp)

    if not flag or min_cost == 1e8:
        return None

    return min_cost


def lowest_cost(recipes, food_item, forbidden=None):
    """
    Given a recipes list and the name of a food item, return the lowest cost of
    a full recipe for the given food item.
    """
    new_recipes = copy.deepcopy(recipes)
    if forbidden is not None:
        for i in forbidden:
            new_recipes = replace_item(new_recipes, i, "null")

    return helper_lowest_cost(new_recipes, food_item)


def combine(x, y):
    res = {}
    for key, value in x.items():
        res[key] = value
    for key, value in y.items():
        if key in res:
            res[key] += value
        else:
            res[key] = value

    return res


def scale(x, sc):
    res = {}
    for key, value in x.items():
        res[key] = value * sc

    return res


# 返回最便宜的食谱的字典
def helper_cheapest_flat_recipe(recipes, food_item):
    res = {}
    if food_item == "null":
        return None
    min_cost = 1e8
    flag = False
    for recipe in recipes:
        if recipe[1] == food_item:
            flag = True
            if recipe[0] == 'atomic':
                return {food_item: 1}
            else:
                temp = 0
                temp_dict = {}
                child_is_none = False
                for i in recipe[2]:
                    lowest_cost_child = helper_lowest_cost(recipes, i[0])
                    if lowest_cost_child is None:
                        child_is_none = True
                        break
                    temp += lowest_cost_child * i[1]
                    temp_dict = combine(temp_dict, scale(helper_cheapest_flat_recipe(recipes, i[0]), i[1]))
                if not child_is_none:
                    if temp < min_cost:
                        min_cost = temp
                        res = temp_dict

    if not flag or min_cost == 1e8:
        return None

    return res



def cheapest_flat_recipe(recipes, food_item, forbidden=None):
    """
    Given a recipes list and the name of a food item, return a dictionary
    (mapping atomic food items to quantities) representing a full recipe for
    the given food item.
    """
    new_recipes = copy.deepcopy(recipes)
    if forbidden is not None:
        for i in forbidden:
            new_recipes = replace_item(new_recipes, i, "null")

    return helper_cheapest_flat_recipe(new_recipes, food_item)


# 返回制作某一食物的所有食谱  字典列表
def helper_all_flat_recipes(recipes, food_item):
    res = []
    if food_item == "null":
        return None
    flag = False
    for recipe in recipes:
        if recipe[1] == food_item:
            flag = True
            if recipe[0] == 'atomic':
                return [{food_item: 1}]
            else:
                temp_array_dict = []    # 遍历到这一层已有的食谱列表
                child_is_none = False
                for i in recipe[2]:
                    lowest_cost_child = helper_lowest_cost(recipes, i[0])
                    if lowest_cost_child is None:
                        child_is_none = True
                        break
                    child_array_dist = helper_all_flat_recipes(recipes, i[0])   # 制作 i 的食谱
                    for c in range(len(child_array_dist)):
                        child_array_dist[c] = scale(child_array_dist[c], i[1])
                    if len(temp_array_dict) == 0:
                        temp_array_dict = child_array_dist.copy()
                    else:
                        new_temp = []
                        for temp in temp_array_dict:
                            for child in child_array_dist:
                                new = combine(temp, child)
                                new_temp.append(new)
                        temp_array_dict = new_temp

                if not child_is_none:
                    res = res + temp_array_dict

    if not flag:
        return []

    return res


def all_flat_recipes(recipes, food_item, forbidden=None):
    """
    Given a list of recipes and the name of a food item, produce a list (in any
    order) of all possible flat recipes for that category.
    """
    new_recipes = copy.deepcopy(recipes)
    if forbidden is not None:
        for i in forbidden:
            new_recipes = replace_item(new_recipes, i, "null")

    return helper_all_flat_recipes(new_recipes, food_item)


if __name__ == "__main__":
    # you are free to add additional testing code here!
   pass
