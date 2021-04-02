import pickle, os
import numpy as np


# class Query_Instance(object):
#     def __init__(self, node1, rel, node2, mode, a_type):
#         self.mode = mode
#         self.interaction_limit = 0
#         self.q_ent = node1
#         self.q_rel = rel
#         self.ans_set = {node2}
#         self.a_type = a_type
#
#     def set_interaction_limit(self, interaction_lt):
#         self.interaction_limit = interaction_lt
#
#     def add_to_answers(self, node):
#         self.ans_set.add(node)


def get_query_instance(node1, rel, node2, mode, interaction_lt):
    head_q_instance = {'q_ent': node2, 'q_rel': rel, 'ans_set': {node1},
                       'a_type': 'h', 'mode': mode, 'interaction_limit': interaction_lt}
    tail_q_instance = {'q_ent': node1, 'q_rel': rel, 'ans_set': {node2},
                       'a_type': 't', 'mode': mode, 'interaction_limit': interaction_lt}
    return head_q_instance, tail_q_instance


def add_to_answers(q_instance, node):
    q_instance['ans_set'].add(node)
    return q_instance


class KB_graph(object):

    def __init__(self, KB_name, path, v_prob):
        self.rel_index = {}
        self.rel_src_nodeset = {}
        self.rel_target_nodeset = {}
        self.node_index = {}
        self.entity_vocab = {}
        self.inv_entity_vocab = {}
        self.inv_rel_vocab = {}
        self.rel_vocab = {}
        self.train_data = {}
        self.rel_domains = {}
        self.domain_entity_set = {}
        self.valid_data = {}
        if KB_name == 'Nell_':
           self.is_domain = True
        else:
           self.is_domain = False
        self.num_edges = 0
        self.valid_prob = v_prob
        print ('path: ', path)
        self.init_KB_graph(path)
        self.KB_name=KB_name

    def init_KB_graph(self, path):
        self.rel_vocab['@-RelatedTo-@'] = len(self.rel_vocab)
        self.inv_rel_vocab[len(self.rel_vocab)-1] = '@-RelatedTo-@'

        self.entity_vocab['@-null-@'] = len(self.entity_vocab)
        self.inv_entity_vocab[len(self.entity_vocab)-1] = '@-null-@'

        base_KB_edges = set()
        file1 = open(path, 'r')
        i = 0
        for line in file1.readlines():
            str1 = line.decode('utf-8').split('\t')
            if len(str1) == 3:
                # print str1[0], str1[1], str1[2], '\n'
                if i % 10000 == 0:
                    print(i)
                node1 = str1[0].replace('\n', '').strip()
                node2 = str1[1].replace('\n', '').strip()
                rel = str1[2].replace('\n', '').strip()

                if self.is_domain:
                    node1_domain = node1.split(':')[0].strip() + ':' + node1.split(':')[1].strip()
                    node2_domain = node2.split(':')[0].strip() + ':' + node2.split(':')[1].strip()
                else:
                    node1_domain = '-'
                    node2_domain = '-'

                # swap triple for inverse relations.
                if rel.endswith('-inv'):
                      rel = rel[:-4]
                      temp = node1
                      node1 = node2
                      node2 = temp

                base_KB_edges.add((node1, node2, rel))
                self.update_KB_all(node1, node2, rel)
                self.add_to_train_eval_set(node1, node2, rel)
                self.update_domain_info(node1_domain, node2_domain, node1, node2, rel)

                i += 1
        print ('Number of edges Read: ', i)
        print ('base KB size:', len(base_KB_edges))
        print ('node size: ', len(self.entity_vocab))
        # print 'Train: ', [(data_key, len(self.train_data[data_key])) for data_key in self.train_data if data_key.endswith('-R')]
        # print 'Valid:', [(data_key, len(self.valid_data[data_key])) for data_key in self.valid_data if data_key.endswith('-R')]
        self.num_edges = len(base_KB_edges)

    def update_domain_info(self, node1_domain, node2_domain, node1, node2, rel):
        if rel in self.rel_domains:
            self.rel_domains[rel]['s_dom'].add(node1_domain)
            self.rel_domains[rel]['t_dom'].add(node2_domain)
        else:
            self.rel_domains[rel] = {}
            self.rel_domains[rel]['s_dom'] = {node1_domain}
            self.rel_domains[rel]['t_dom'] = {node2_domain}

        if node1_domain in self.domain_entity_set:
            self.domain_entity_set[node1_domain].add(node1)
        else:
            self.domain_entity_set[node1_domain] = {node1}

        if node2_domain in self.domain_entity_set:
            self.domain_entity_set[node2_domain].add(node2)
        else:
            self.domain_entity_set[node2_domain] = {node2}

    def initialize_dataset(self, node1, rel, node2, data):
        # update in rel part
        if rel + '-R' not in data:
            data[rel + '-R'] = set()
        # update in source part
        if node1 + '-ENT' not in data:
            data[node1 + '-ENT'] = {}
        # update in target part
        if node2 + '-ENT' not in data:
            data[node2 + '-ENT'] = {}
        if node1 + '-NDX' not in data:
            data[node1 + '-NDX'] = set()
        if node2 + '-NDX' not in data:
            data[node2 + '-NDX'] = set()

    def add_to_train_eval_set(self, node1, node2, rel):

        data = self.train_data
        mode = 'T'
        if rel+'-R' in self.train_data and node1+'-ENT' in self.train_data and node2+'-ENT' in self.train_data:
            num_tr_ex = len(self.train_data[rel + '-R'])

            if rel + '-R' in self.valid_data:
                num_vd_ex = len(self.valid_data[rel + '-R'])
            else:
                num_vd_ex = 0
                data = self.valid_data
                mode = 'V'

            if num_tr_ex > 0 and num_vd_ex > 0:
                vd_tr_ratio = (num_vd_ex * 1.0) / num_tr_ex

                if vd_tr_ratio < (self.valid_prob/(1 - self.valid_prob)):
                    data = self.valid_data
                    mode = 'V'
                    #print mode, vd_tr_ratio, num_vd_ex, num_tr_ex, rel

        if mode == 'V' and rel + '-R' in self.train_data and (node1, rel, node2) in self.train_data[rel + '-R']:
            return
        if mode == 'T' and rel + '-R' in self.valid_data and (node1, rel, node2) in self.valid_data[rel + '-R']:
            return

        self.initialize_dataset(node1, rel, node2, data)
        head_q_instance, tail_q_instance = get_query_instance(node1, rel, node2, mode, 0)

        # rel ---> node_pairs
        data[rel + '-R'].add((node1, rel, node2))
        data[node1 + '-NDX'].add((node1, rel, node2))
        data[node2 + '-NDX'].add((node1, rel, node2))

        # update for node1
        if node1 + "#" + rel + '#-' in data[node1+'-ENT']:
            q_instance = data[node1+'-ENT'][node1 + "#" + rel + '#-']
            add_to_answers(q_instance, node2)
            data[node1 + '-ENT'][node1 + "#" + rel + '#-'] = q_instance
        else:
            data[node1+'-ENT'][node1 + "#" + rel + '#-'] = tail_q_instance

        if "-#" + rel + "#" + node2 in data[node1+'-ENT']:
            q_instance = data[node1+'-ENT']["-#" + rel + "#" + node2]
            add_to_answers(q_instance, node1)
            data[node1 + '-ENT']["-#" + rel + "#" + node2] = q_instance
        else:
            data[node1 + '-ENT']["-#" + rel + "#"+ node2] = head_q_instance

        # update for node2
        if node1 + "#" + rel + '#-' in data[node2+'-ENT']:
            q_instance = data[node1+'-ENT'][node1 + "#" + rel + '#-']
            add_to_answers(q_instance, node2)
            data[node2 + '-ENT'][node1 + "#" + rel + '#-'] = q_instance
        else:
            data[node2+'-ENT'][node1 + "#" + rel + '#-'] = tail_q_instance

        if "-#" + rel + "#" + node2 in data[node2 + '-ENT']:
            q_instance = data[node1 + '-ENT']["-#" + rel + "#" + node2]
            add_to_answers(q_instance, node1)
            data[node2 + '-ENT']["-#" + rel + "#" + node2] = q_instance
        else:
            data[node2 + '-ENT']["-#" + rel + "#" + node2] = head_q_instance

    def update_KB_all(self, node1, node2, rel):
        self.update_node_index(node1, node2, rel)
        self.update_rel_index(node1, node2, rel)
        self.update_KB_vocab(node1, node2, rel)

    def update_node_index(self, node1, node2, rel):
        pass

        # For Source node ...
        # if self.node_index.has_key(node1):
        #     node_rel_pairs = self.node_index[node1]
        #
        #     if node_rel_pairs.has_key(node2):
        #         if rel not in node_rel_pairs[node2]:
        #             node_rel_pairs[node2].append(rel)
        #     else:
        #         node_rel_pairs[node2] = [rel]
        #     self.node_index[node1] = node_rel_pairs
        # else:
        #     node_rel_pairs = {node2: [rel]}
        #     self.node_index[node1] = node_rel_pairs
        #
        # # For Target Node ...
        # if self.node_index.has_key(node2):
        #     node_rel_pairs = self.node_index[node2]
        #
        #     if node_rel_pairs.has_key(node1):
        #         if rel + '-inv' not in node_rel_pairs[node1]:
        #             node_rel_pairs[node1].append(rel + '-inv')
        #     else:
        #         node_rel_pairs[node1] = [rel + '-inv']
        #     self.node_index[node1] = node_rel_pairs
        # else:
        #     node_rel_pairs = {node1: [rel + '-inv']}
        #     self.node_index[node2] = node_rel_pairs

    def update_rel_index(self, node1, node2, rel):
        if self.rel_index.has_key(rel):
           self.rel_index[rel].add(node1+'-;-'+node2)
        else:
           self.rel_index[rel] = {node1+'-;-'+node2}

        if rel in self.rel_src_nodeset:
           self.rel_src_nodeset[rel].add(node1)
        else:
           self.rel_src_nodeset[rel] = {node1}

        if rel in self.rel_target_nodeset:
           self.rel_target_nodeset[rel].add(node2)
        else:
           self.rel_target_nodeset[rel] = {node2}

    def update_KB_vocab(self, node1, node2, rel):
        if not self.entity_vocab.has_key(node1):
            self.entity_vocab[node1] = len(self.entity_vocab)
            self.inv_entity_vocab[self.entity_vocab[node1]] = node1
            #print node1, 'added'

        if not self.entity_vocab.has_key(node2):
            self.entity_vocab[node2] = len(self.entity_vocab)
            self.inv_entity_vocab[self.entity_vocab[node2]] = node2
            #print node2, 'added'

        if not self.rel_vocab.has_key(rel):
            self.rel_vocab[rel] = len(self.rel_vocab)
            self.inv_rel_vocab[self.rel_vocab[rel]] = rel


def load_KB(KB_name, path, valid_prob):
    if os.path.isfile('./KB_dumps/' + KB_name + "0_KB_dump.pickle"):
        with open('./KB_dumps/' + KB_name + "0_KB_dump.pickle", "rb") as input_file:
            KB = pickle.load(input_file)
    else:
        KB = KB_graph(KB_name, path, valid_prob)
        if not os.path.isdir('./KB_dumps/'):
            os.makedirs('./KB_dumps/')
        with open('./KB_dumps/' + KB_name + '0_KB_dump.pickle', 'w') as graph_file:
            pickle.dump(KB, graph_file)
    return KB