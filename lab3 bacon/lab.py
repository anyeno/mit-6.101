#!/usr/bin/env python3

import pickle

# NO ADDITIONAL IMPORTS ALLOWED!
import queue


def transform_data(raw_data):
    res = {}
    for i in raw_data:
        res.setdefault(i[0], []).append((i[1], i[2]))
        res.setdefault(i[1], []).append((i[0], i[2]))
    # print(res)

    return res


def acted_together(transformed_data, actor_id_1, actor_id_2):
    if actor_id_1 == actor_id_2:
        return True
    for i in transformed_data[actor_id_1]:
        if i[0] == actor_id_2:
            return True
    return False


def actors_with_bacon_number(transformed_data, n):
    actor2bacon = {}
    q = queue.Queue()
    q.put(4724)
    actor2bacon[4724] = 0
    res = set()
    while not q.empty():
        t = q.get()
        if actor2bacon[t] == n:
            res.add(t)
        if actor2bacon[t] > n:
            break
        for i in transformed_data[t]:
            j = i[0]
            if j not in actor2bacon:
                actor2bacon[j] = actor2bacon[t] + 1
                q.put(j)
    if n == 1:
        res.add(4724)

    return res


def bacon_path(transformed_data, actor_id):
    return actor_to_actor_path(transformed_data, 4724, actor_id)


def actor_to_actor_path(transformed_data, actor_id_1, actor_id_2):
    if actor_id_1 not in transformed_data or actor_id_2 not in transformed_data:
        return None

    pre = {}  # 路径上的前一个点
    q = queue.Queue()
    q.put(actor_id_2)
    flag = False
    while not q.empty():
        t = q.get()
        if t == actor_id_1:
            flag = True
            break
        for i in transformed_data[t]:
            j = i[0]
            if not j in pre:
                pre[j] = t
                q.put(j)

    if flag == False:
        return None
    st = actor_id_1
    res = []
    while st != actor_id_2:
        res.append(st)
        print(st)
        st = pre[st]
    res.append(actor_id_2)

    return res


def actor_path(transformed_data, actor_id_1, goal_test_function):
    if actor_id_1 not in transformed_data:
        return None
    res = []
    pre = {}
    pre[actor_id_1] = -1
    q = queue.Queue()
    q.put(actor_id_1)
    end = -1
    flag = False
    while not q.empty():
        t = q.get()
        if goal_test_function(t):
            end = t
            flag = True
            break
        for i in transformed_data[t]:
            j = i[0]
            if not j in pre:
                pre[j] = t
                q.put(j)
    if flag == False:
        return None
    res.append(end)
    while end != actor_id_1:
        end = pre[end]
        res.insert(0, end)
    print(res)
    return res


def helper_acf_goal(transformed_data, film2):
    def goal_test(actor_id):
        for i in transformed_data[actor_id]:
            if i[1] == film2:
                return True
        return False
    return goal_test

def actors_connecting_films(transformed_data, film1, film2):
    res = []
    min_len = 1e8
    st = set()
    goal_test_function = helper_acf_goal(transformed_data, film2)
    for k, v in transformed_data.items():
        for j in v:
            if j[1] == film1:
                if not j[1] in st:
                    st.add(j[1])
                    res_in =  actor_path(transformed_data, k, goal_test_function)
                    if len(res_in) < min_len:
                        min_len = len(res_in)
                        res = res_in
    print(res)
    return res

if __name__ == "__main__":
    with open("resources/small.pickle", "rb") as f:
        smalldb = pickle.load(f)

    # additional code here will be run only when lab.py is invoked directly
    # (not when imported from test.py), so this is a good place to put code
    # used, for example, to generate the results for the online questions.

    res = transform_data(smalldb)
