import tensorflow as tf
print(tf.__version__)
import os
from controller_module_5 import *
from DistMult_5 import *
import random
import pickle as pickle2
from batch_training_helper import *
from util_helper import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(1000)

coverage_stat={}
coverage_stat_filtered={}
action_stat={}
rList = []
epi_len_list=[]


class agent():
    def __init__(self, is_load):
        self.controller = None
        self.is_load = is_load
        self.result_folder = ''
        print ('Initializing dec_agent ...')

    def initialize_controller(self):
        self.controller = Controller()

        # Setting up KB ....
        self.KB = load_KB(self.controller.KB_name, self.controller.KB_path, self.controller.valid_prob)
        print ('rel Vocab: ', sorted(self.KB.rel_vocab.items(), key=lambda x: x[1]))
        print (len(self.KB.rel_vocab))
        print ('Train: ', [(rel, len(self.KB.train_data[rel])) for rel in self.KB.train_data if rel.endswith('-R')])
        print ('Valid:', [(rel, len(self.KB.valid_data[rel])) for rel in self.KB.valid_data if rel.endswith('-R')])

        print ('Train: ', np.sum([len(self.KB.train_data[rel]) for rel in self.KB.train_data if rel.endswith('-R')]))
        print ('Valid:', np.sum([len(self.KB.valid_data[rel]) for rel in self.KB.valid_data if rel.endswith('-R')]))

        all_train_triples = []
        for rel in self.KB.train_data:
            if rel.endswith('-R'):
                all_train_triples.extend(self.KB.train_data[rel])

        all_valid_triples = []
        for rel in self.KB.valid_data:
            if rel.endswith('-R'):
                all_valid_triples.extend(self.KB.valid_data[rel])

        print ('train_triples: ', len(all_train_triples))
        print ('valid_triples: ', len(all_valid_triples))

        print (len(set(all_train_triples).intersection(set(all_valid_triples))))


        # DATA SET STATS
        print ('------------------------------------------')
        print ('========   TRAIN DATA STATS ===============')
        print ('------------------------------------------')
        get_dataset_stats(self.KB.train_data, self, 'train')
        print ('------------------------------------------')
        print ('========   VALID DATA STATS ===============')
        print ('------------------------------------------')
        get_dataset_stats(self.KB.valid_data, self, 'valid')

        # Setting up Inference module ...
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.14)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options))
        self.distmult_model = DistMult(self.sess, self.controller.max_rel_vocab_size, self.controller.max_ent_vocab_size,
                                              self.controller.neg_samples_per_triple, self.controller.embd_dim, lamda=self.controller.lamda)
        self.saver = tf.compat.v1.train.Saver()
        load_or_initialize_model(self.sess, self.saver, self.controller.KB_name,
                                 '../resource_5/' + self.controller.KB_name + '_DistMult_model/')

        # simulated user ...
        self.user = load_Simulated_User(self.controller)
        self.evaluation_buffer = []
        self.known_info_map = {'rel': set(), 'ent': set()}
        self.unknown_rel_set = set()

        self.performance_buff, self.thresh_buff = load_or_initialize_buffers(self.controller.KB_name)
        print ('dec_agent initialized ...starting training ...')

    def initial_training(self):
        self.controller.PLE_rel_set, self.controller.PLE_ent_set = self.controller.get_PLE_sets(self.performance_buff)

        if os.path.isfile('./KB_dumps/' + self.controller.KB_name + "0_init_dataset.pickle"):
            with open('./KB_dumps/' + self.controller.KB_name + "0_init_dataset.pickle", "rb") as input_file:
                init_datasets = pickle2.load(input_file)

            initial_trainset = init_datasets[0]
            initial_validset = init_datasets[1]
            print ('initial train valid loaded ...')
            print ('init Train / Valid: ', len(initial_trainset), len(initial_validset))
        else:
            # training , valid stream creation...
            train_stream, valid_stream = get_train_valid_stream(self.KB)
            train_stream_len = len(train_stream)
            print ('# train_instances:', train_stream_len)
            print ('# valid_instances:', len(valid_stream))

            print ('generating train set')
            initial_trainset = get_full_batch_dataset(train_stream, self.KB, 'T')
            print (train_stream_len, len(initial_trainset))

            print ('generating validation set')
            initial_validset = get_full_batch_dataset(valid_stream, self.KB, 'V')
            print (train_stream_len, len(initial_trainset))
            print (len(valid_stream), len(initial_validset))

            init_datasets = (initial_trainset, initial_validset)
            with open('./KB_dumps/' + self.controller.KB_name + "0_init_dataset.pickle", 'w') as out_file:
                pickle2.dump(init_datasets, out_file)
            print ('initial train valid dumped ...')

        train_and_evaluate_inf_model_full_batch(initial_trainset, initial_validset, self.distmult_model,
                         self.controller.tr_batch_size, self, max_epoch=self.controller.init_train_epoch)

        # save trained model...
        save_model(self.sess, self.saver, self.controller.KB_name, '../resource_5/' + self.controller.KB_name + '_DistMult_model/')
        save_perf_thresh_buffers(self.controller.KB_name, self.performance_buff, self.thresh_buff)
        print ('dumping perf_buff and thresh_buff ....')

    def run_eval(self):
        print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> AFTER Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ...')
        test_data_store = load_test_dataset(self)
        # print 'unknown_rels: ', self.controller.unknown_rel_set
        # print 'unknown_tr_rels: ', self.controller.unknown_tr_rel_set
        print("NODE LIST SIZE: " + str(len(self.KB.entity_vocab)))
        print("REL LIST SIZE: " + str(len(self.KB.rel_vocab)))
        print("EDGE LIST SIZE: " + str(self.KB.num_edges))
        print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ...')
        self.controller.PLE_rel_set = []
        self.controller.PLE_ent_set = []
        print ('PLE SET: ', self.controller.PLE_rel_set)

        # test stream creation...
        test_stream = []
        for dict_key in test_data_store.keys():
            if dict_key.endswith('-R'):
                test_stream.extend(test_data_store[dict_key])
        # create random cronological ordering ....
        for k in range(10):
            random.shuffle(test_stream)
        test_stream_len = len(test_stream)
        print ('# test_instances:', test_stream_len)

        # testing ...
        self.user.reset_counters()
        self.user.query_list = {}
        self.evaluation_buffer=[]

        q_rel_unk_triples = 0
        q_ent_unk_triples = 0
        ent_rel_unk_triples = 0

        # evaluation started ....
        test_index = 0
        while test_index < test_stream_len:
            self.controller.PLE_rel_set, self.controller.PLE_ent_set = self.controller.get_PLE_sets(self.performance_buff)
            #print 'PLE SET rel during Test: ', self.controller.PLE_rel_set, len(self.controller.PLE_rel_set)

            # detect task
            test_triple = test_stream[test_index]
            node1 = test_triple[0]
            rel = test_triple[1]
            node2 = test_triple[2]

            head_q_test = test_data_store[node1 + '-ENT']["-#" + rel + "#" + node2]
            tail_q_test = test_data_store[node1 + '-ENT'][node1 + "#" + rel + '#-']

            print ('test index: ', test_index+1, test_stream_len)

            if np.random.rand(1)[0] < 0.5:
                if node2 not in self.known_info_map['ent'] and rel in self.known_info_map['rel']:
                    q_ent_unk_triples += 1
                if node2 in self.known_info_map['ent'] and rel not in self.known_info_map['rel']:
                    q_rel_unk_triples += 1
                if node2 not in self.known_info_map['ent'] and rel not in self.known_info_map['rel']:
                    ent_rel_unk_triples += 1

                self.controller.execute_episode(head_q_test, 'eval', self.distmult_model, self)
            else:
                if node1 not in self.known_info_map['ent'] and rel in self.known_info_map['rel']:
                    q_ent_unk_triples += 1
                if node1 in self.known_info_map['ent'] and rel not in self.known_info_map['rel']:
                    q_rel_unk_triples += 1
                if node1 not in self.known_info_map['ent'] and rel not in self.known_info_map['rel']:
                    ent_rel_unk_triples += 1

                self.controller.execute_episode(tail_q_test, 'eval', self.distmult_model, self)

            print ('test_data_index: ', test_index+1, 'eval_buff_size: ', len(self.evaluation_buffer))
            test_index += 1

        print (' Dumping evaluation buffer ...')
        with open('./KB_dumps/'+dec_agent.result_folder+'/' + self.controller.KB_name + 'eval_buff.pickle', 'w') as out_file:
            pickle2.dump(self.evaluation_buffer, out_file)
        with open('./KB_dumps/'+dec_agent.result_folder+'/' + self.controller.KB_name + 'known_info_map.pickle', 'w') as out_file:
            pickle2.dump(self.known_info_map, out_file)

        print ('only ent UNK: ', (q_ent_unk_triples * 100.0) / test_stream_len)
        print ('only rel UNK: ', (q_rel_unk_triples * 100.0) / test_stream_len)
        print ('both ent and rel UNK: ', (ent_rel_unk_triples * 100.0) / test_stream_len)

        print ('All epoch completed ...query log written...')


if __name__ == "__main__" :
    dec_agent = agent(is_load = True)  # Load the agent.

    dec_agent.initialize_controller()
    dec_agent.result_folder = 'bth_btr_c1_ef3'
    if not os.path.isdir('./KB_dumps/'+dec_agent.result_folder +'/'):
        os.makedirs('./KB_dumps/'+dec_agent.result_folder+'/')
    if not dec_agent.is_load:
       dec_agent.initial_training()
    else:
       dec_agent.run_eval()

