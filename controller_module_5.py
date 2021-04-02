from KG_Creation_2 import *
from Inf_model_training import *
from util_helper import *
import copy


class Controller(object):

    def __init__(self):
        print ('Configuring Controller Module...')
        # Graph, vocab initialized ...
        #self.KB_name = 'Freebase_'
        # self.KB_name = 'Conceptnet_'
        #self.KB_name = 'Wordnet_'
        self.KB_name = 'Nell_'

        self.threshold_type = 'both' # 'both', 'ent', 'rel', 'any'
        self.training_mode = 'both' # both, 'rel', 'ent'
        self.PLE_enabled = True
        self.beta = 1.0
        self.neg_samples_per_triple = 2

        print ('Beta: ', self.beta)

        self.set_controller_config()
        print ('Module configured')
        print ('Executor initialized...')

    def set_controller_config(self):
        self.sample_size = 500
        self.nprocs = 5
        self.interaction_lt = 5
        self.verbose = True
        self.tr_batch_size = 128
        self.valid_prob = 0.1
        self.initial_tr_queryset = {}

        path = ''
        if self.KB_name == 'Wordnet_':
            self.max_rel_vocab_size = 60
            self.max_ent_vocab_size = 17000
            self.embd_dim = 250
            self.avg_tr_size = 994
            self.init_train_epoch = 100
            self.lr1 = 0.001
            self.lr2 = 0.001
            self.lamda = 0.001
            self.KB_path = '../resource_5/Wordnet_edgelist_pra0.tsv'
        elif self.KB_name == 'Conceptnet_':
            self.max_rel_vocab_size = 150
            self.max_ent_vocab_size = 100000
            self.embd_dim = 300
            self.avg_tr_size = 1800
            self.KB_path = '../resource_5/Conceptnet_edgelist_pra0.tsv'
        elif self.KB_name == 'Freebase_':
            self.max_rel_vocab_size = 150
            self.max_ent_vocab_size = 25000
            self.embd_dim = 250
            self.avg_tr_size = 1685
            self.init_train_epoch = 102
            self.lr1 = 0.001
            self.lr2 = 0.001
            self.lamda = 0.001
            self.KB_path = '../resource_5/Freebase_edgelist_pra0.tsv'
        elif self.KB_name == 'Nell_':
            self.max_rel_vocab_size = 150
            self.max_ent_vocab_size = 17000
            self.embd_dim = 250
            self.avg_tr_size = 1685
            self.init_train_epoch = 140
            self.lr1 = 0.001
            self.lr2 = 0.001
            self.lamda = 0.001
            self.KB_path = '../resource_5/Nell_edgelist_pra0.tsv'

        self.PLE_rel_set = None
        self.PLE_ent_set = None

    def get_PLE_sets(self, perf_buff):
        if self.PLE_enabled:
            candidate_set = copy.deepcopy(perf_buff['rel'])
            rel_set = sorted(candidate_set.items(), key=itemgetter(1))
            #print 'Full rel_set: ', rel_set, len(rel_set)
            rel_set=rel_set[:int(len(candidate_set) * 0.2)]
            #print 'Low mcc rels: ', rel_set

            candidate_set = copy.deepcopy(perf_buff['ent'])
            ent_set = sorted(candidate_set.items(), key=itemgetter(1))
            ent_set = ent_set[:int(len(candidate_set) * 0.2)]
            # print 'Low mcc rels: ', rel_set

            return set([rel[0] for rel in rel_set]), set([ent[0] for ent in ent_set])
        else:
            return {}, {}

    def search_source_cpt_and_query_rel(self, source, rel, KB):
        return KB.entity_vocab.has_key(source), KB.rel_vocab.has_key(rel)

    def update_KB(self, KB, clue_source, clue_target, rel):
        if KB.is_domain:
            node1_domain = clue_source.split(':')[0].strip() + ':' + clue_source.split(':')[1].strip()
            node2_domain = clue_target.split(':')[0].strip() + ':' + clue_target.split(':')[1].strip()
        else:
            node1_domain = '-'
            node2_domain = '-'

        KB.update_KB_all(clue_source, clue_target, rel)
        KB.add_to_train_eval_set(clue_source, clue_target, rel)
        KB.update_domain_info(node1_domain, node2_domain, clue_source, clue_target, rel)

    def execute_episode(self, test_q, phase, distmult_model, agent):

        # search in KB
        q_ent = test_q['q_ent']
        q_rel = test_q['q_rel']
        res_q_ent, res_q_rel = self.search_source_cpt_and_query_rel(q_ent, q_rel, agent.KB)
        #print 'search res: ', res_q_ent, res_q_rel
        inter_flag = False

        # Ask for clue:
        clue_set = {}
        if ((not res_q_rel) or (q_rel in self.PLE_rel_set)) and test_q['interaction_limit'] > 0:
            test_q['interaction_limit'] -= 1
            clue_set = agent.user.Ask_Simulated_User_For_Example(q_rel)
            inter_flag = True

        # Ask for query entity facts..
        ent_fact_set = {}
        if ((not res_q_ent) or (q_ent in self.PLE_ent_set)) and test_q['interaction_limit'] > 0:
            test_q['interaction_limit'] -= 1
            ent_fact_set = agent.user.Ask_Simulated_User_For_connecting_link(q_ent)
            inter_flag = True

        # update KB with clues
        if len(clue_set) > 0:
            for clue_source, clue_target in clue_set:
                self.update_KB(agent.KB, clue_source, clue_target, q_rel)

        # update KB with ent_facts
        if len(ent_fact_set) > 0:
            for s1, link, t1 in ent_fact_set:
                # swap triple for inverse relations.
                if link.endswith('-inv'):
                    link = link[:-4]
                    temp = s1
                    s1 = t1
                    t1 = temp
                self.update_KB(agent.KB, s1, t1, link)

        res_q_ent2, res_q_rel2 = self.search_source_cpt_and_query_rel(q_ent, q_rel, agent.KB)
        if res_q_ent2 and res_q_rel2:
            # train / inference
            if res_q_ent and res_q_rel:
                train_and_evaluate_inf_model(q_ent, q_rel, distmult_model, self.tr_batch_size, agent, 5, self.lr1)
            else:
                train_and_evaluate_inf_model(q_ent, q_rel, distmult_model, self.tr_batch_size, agent, 2, self.lr1)

            # option 2
            # if res_q_ent and res_q_rel and inter_flag:
            #     train_and_evaluate_inf_model(q_ent, q_rel, distmult_model, self.tr_batch_size, agent, 5, self.lr1)
            # else:
            #     train_and_evaluate_inf_model(q_ent, q_rel, distmult_model, self.tr_batch_size, agent, 2, self.lr1)

            if test_q['mode'] == 'E' and phase == 'eval':
                run_inference(test_q, distmult_model, agent)
