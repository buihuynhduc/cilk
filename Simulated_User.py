# uncompyle6 version 3.5.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.5 (default, Aug  7 2019, 00:51:29)
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-39)]
# Embedded file name: /home/user/test_cle/code_repo/Simulated_User.py
# Compiled at: 2019-07-28 06:08:09
import random, requests

def init_simulated_user(path, user_node_index, user_rel_index):
    rel_file = open(path, 'r').readlines()
    print ('User KB rels.: ', len(rel_file))
    for i in range(0, len(rel_file), 1):
        rel = rel_file[i].decode('utf-8').split('--->')[0]
        rel = rel.replace('/', '_')
        nodes = rel_file[i].decode('utf-8').split('--->')[1].split('##')
        nodes.remove(nodes[(len(nodes) - 1)])
        if user_rel_index.has_key(rel):
            user_rel_index[rel].extend(nodes)
        else:
            user_rel_index[rel] = nodes
        for j in range(0, len(nodes), 1):
            source = nodes[j].split('-;-')[0].strip()
            target = nodes[j].split('-;-')[1].strip()
            if user_node_index.has_key(source):
                node_rel_pairs = user_node_index[source]
                if node_rel_pairs.has_key(target):
                    if rel not in node_rel_pairs[target]:
                        node_rel_pairs[target].append(rel)
                else:
                    node_rel_pairs[target] = [
                     rel]
                user_node_index[source] = node_rel_pairs
            else:
                node_rel_pairs = {target: [rel]}
                user_node_index[source] = node_rel_pairs
            if user_node_index.has_key(target):
                node_rel_pairs = user_node_index[target]
                if node_rel_pairs.has_key(source):
                    if rel + '-inv' not in node_rel_pairs[source]:
                        node_rel_pairs[source].append(rel + '-inv')
                else:
                    node_rel_pairs[source] = [
                     rel + '-inv']
                user_node_index[target] = node_rel_pairs
            else:
                node_rel_pairs = {source: [rel + '-inv']}
                user_node_index[target] = node_rel_pairs


class Simulated_User(object):
    user_rel_index = {}
    user_node_index = {}

    def __init__(self, path, KB):
        init_simulated_user(path, self.user_node_index, self.user_rel_index)
        self.KB = KB
        self.number_clues_asked = 0
        self.number_clues_answered = 0
        self.number_conn_link_query_asked = 0
        self.number_conn_link_query_answered = 0
        self.query_list = {}
        self.user_response_prob = 1.0
        print ('user interaction: ', self.user_response_prob)

    def reset_counters(self):
        self.number_clues_asked = 0
        self.number_clues_answered = 0
        self.number_conn_link_query_asked = 0
        self.number_conn_link_query_answered = 0

    def Ask_Simulated_User_For_Example(self, rel, num_clues=1):
        self.number_clues_asked += 1
        if self.user_rel_index.has_key(rel) and len(self.user_rel_index[rel]) > 0:
            clue_set = random.sample(self.user_rel_index[rel], min(self.user_rel_index[rel], num_clues))
            self.number_clues_answered += 1
            return {(example.split('-;-')[0].strip(), example.split('-;-')[1].strip()) for example in clue_set}
        else:
            return {}

    def Ask_Simulated_User_For_connecting_link(self, unk_cpt, num_facts=3):
        self.query_list[unk_cpt] = ''
        self.number_conn_link_query_asked += 1
        response_set = set()
        if self.user_node_index.has_key(unk_cpt):
            node_rel_pairs = self.user_node_index[unk_cpt]
            target_nodes = random.sample(node_rel_pairs.keys(), min(len(node_rel_pairs.keys()), num_facts))
            for t_node in target_nodes:
                link = random.choice(self.user_node_index[unk_cpt][t_node])
                response_set.add((unk_cpt, link, t_node))

            self.number_conn_link_query_answered += 1
        return response_set


if __name__ == '__main__':
    path = './KB_dumps/user_wordNet0.txt'
    user = Simulated_User(path)