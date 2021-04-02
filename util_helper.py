
import numpy as np, os, pickle
from Simulated_User import *
import pickle as pickle2, math
from Simulated_User import Simulated_User
import tensorflow as tf, time
from KG_Creation_2 import *
from multiprocessing import Process, Queue


def pad_arr_seq(curr_seq, max_len, padding_seq):
    #assert len(padding_seq) == len(curr_seq[0])
    for i in range(max_len -len(curr_seq)):
         curr_seq.insert(0, padding_seq)
    return curr_seq


def get_processing_time(start_time):
    return round((time.time() - start_time)/60.0, 2)


def load_Simulated_User(controller):
    # Simulated user instantiated ....
    path = None
    if controller.KB_name == 'Wordnet_':
        path = '../resource_5/user_wordNet0.txt'
    elif controller.KB_name == 'Freebase_':
        path = '../resource_5/user_freebase0.txt'
    elif controller.KB_name == 'Conceptnet_':
        path = '../resource_5/user_conceptnet0.txt'
    elif controller.KB_name == 'Nell_':
        path = '../resource_5/user_nell.txt'
    return Simulated_User(path, controller.KB_name)


def load_or_initialize_model(sess, saver, model_name, model_path):
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(model_path+model_name+'.ckpt.meta'):
        saver.restore(sess, model_path+model_name+'.ckpt')
        print(model_name+" Model restored.")
        return True
    else:
        print(model_name+ " Model initialized.")
        return False


def save_model(sess, saver, model_name, model_path):
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    # Save model weights to disk
    save_path = saver.save(sess, model_path+model_name+".ckpt")
    print(" Model saved in file: %s at episode:" % save_path)


def load_test_dataset(agent):
    test_path = None

    if agent.controller.KB_name == 'Wordnet_':
        test_path = '../resource_5/test_wordNet0.txt'
    elif agent.controller.KB_name == 'Freebase_':
        test_path = '../resource_5/test_freebase0.txt'
    elif agent.controller.KB_name == 'Conceptnet_':
        test_path = '../resource_5/test_conceptnet0.txt'
    elif agent.controller.KB_name == 'Nell_':
        test_path = '../resource_5/test_nell.txt'

    rel_file = open(test_path, 'r').readlines()
    print (len(rel_file))
    data = {}

    for i in range(0, len(rel_file), 1):
        rel = rel_file[i].decode('utf-8').split('--->')[0]
        nodes = rel_file[i].decode('utf-8').split('--->')[1].split('##')
        nodes.remove(nodes[len(nodes) - 1])

        if rel + '-R' not in data:
            data[rel + '-R'] = set()

        # positive examples ...
        for i in range(0, int(len(nodes)), 1):
            # print rel, "$$$$$$$$$$$$$$$$ ===========>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", i

            node1 = nodes[i].split('-;-')[0].strip()
            node2 = nodes[i].split('-;-')[1].strip()

            data[rel + '-R'].add((node1, rel, node2))

            # update in source part
            if node1 + '-ENT' not in data:
                data[node1 + '-ENT'] = {}
            # update in target part
            if node2 + '-ENT' not in data:
                data[node2 + '-ENT'] = {}

            head_q_instance, tail_q_instance = get_query_instance(node1, rel, node2, 'E', agent.controller.interaction_lt)

            # update for node1
            if node1 + "#" + rel + '#-' in data[node1 + '-ENT']:
                q_instance = data[node1 + '-ENT'][node1 + "#" + rel + '#-']
                add_to_answers(q_instance, node2)
                data[node1 + '-ENT'][node1 + "#" + rel + '#-'] = q_instance
            else:
                data[node1 + '-ENT'][node1 + "#" + rel + '#-'] = tail_q_instance

            if "-#" + rel + "#" + node2 in data[node1 + '-ENT']:
                q_instance = data[node1 + '-ENT']["-#" + rel + "#" + node2]
                add_to_answers(q_instance, node2)
                data[node1 + '-ENT'][node1 + "#" + rel + '#-'] = q_instance
            else:
                data[node1 + '-ENT']["-#" + rel + "#" + node2] = head_q_instance

            # update for node2
            if node1 + "#" + rel + '#-' in data[node2 + '-ENT']:
                q_instance = data[node1 + '-ENT'][node1 + "#" + rel + '#-']
                add_to_answers(q_instance, node2)
                data[node2 + '-ENT'][node1 + "#" + rel + '#-'] = q_instance
            else:
                data[node2 + '-ENT'][node1 + "#" + rel + '#-'] = tail_q_instance

            if "-#" + rel + "#" + node2 in data[node2 + '-ENT']:
                q_instance = data[node1 + '-ENT']["-#" + rel + "#" + node2]
                add_to_answers(q_instance, node2)
                data[node2 + '-ENT'][node1 + "#" + rel + '#-'] = q_instance
            else:
                data[node2 + '-ENT']["-#" + rel + "#" + node2] = head_q_instance

        print (len(data[rel + '-R']))
    print ('test Dataset loaded...!')

    print ('------------------------------------------')
    print ('========   TEST DATA STATS ===============')
    print ('------------------------------------------')
    get_dataset_stats(data, agent, 'test')
    return data


def get_dataset_stats(data, agent, type):
    print ('REL --->'.ljust(
        30), '\t\t', 'Unknown Source (%)', '\t', 'Unknown Target (%)', '\t', 'Unknown Source-Target (%)', \
        '\t', 'Known Source-Target (%)', '\t', 'Unknown Rel (%)')
    unk_source = []
    unk_target = []
    unk_source_target = []
    known_source_target = []

    for rel in data.keys():
        if rel.endswith('-R'):
            total_instance = 0
            source_unknown_instance = 0
            target_unknown_instance = 0
            both_entity_unknown_instance = 0
            both_entity_known_instance = 0
            rel_unknown_instance = 0

            for triple in data[rel]:

                source_exists =triple[0] in agent.KB.entity_vocab
                rel_exists =triple[1] in agent.KB.rel_vocab
                target_exists =triple[2] in agent.KB.entity_vocab
                total_instance += 1

                if triple[1] in agent.KB.train_data and triple in agent.KB.train_data[triple[1] + '-R']:
                    print ('test triple found!')

                if type == 'test':
                    if source_exists:
                        agent.known_info_map['ent'].add(triple[0])
                    if target_exists:
                        agent.known_info_map['ent'].add(triple[2])
                    if rel_exists:
                        agent.known_info_map['rel'].add(triple[1])

                if not source_exists and target_exists:
                    source_unknown_instance += 1
                if source_exists and not target_exists:
                    target_unknown_instance += 1
                if not source_exists and not target_exists:
                    both_entity_unknown_instance += 1
                if source_exists and target_exists:
                    both_entity_known_instance += 1
                if not rel_exists:
                    rel_unknown_instance += 1

            unk_source.append(((source_unknown_instance * 100.0) / total_instance))
            unk_target.append(((target_unknown_instance * 100.0) / total_instance))
            unk_source_target.append(((both_entity_unknown_instance * 100.0) / total_instance))
            known_source_target.append(((both_entity_known_instance * 100.0) / total_instance))

            print (rel.ljust(30), '---->', '\t\t\t', '%.3f' % ((source_unknown_instance * 100.0) / total_instance), '\t\t\t', \
                '%.3f' % ((target_unknown_instance * 100.0) / total_instance), '\t\t\t', \
                '%.3f' % ((both_entity_unknown_instance * 100.0) / total_instance), '\t\t\t', \
                '%.3f' % ((both_entity_known_instance * 100.0) / total_instance), '\t\t\t', \
                '%.3f' % ((rel_unknown_instance * 100.0) / total_instance))

            if rel_unknown_instance > 0:
                agent.unknown_rel_set.add(rel)

                # if type == 'train':
                #     agent.unknown_tr_rel_set.add(rel)

    print ('**********************************')
    # print 'Avg. % source unknown: ', np.mean(unk_source)
    # print 'Avg. % target unknown: ', np.mean(unk_target)
    # print 'Avg. % source target unknown: ', np.mean(unk_source_target)
    # print 'Avg. % source target known: ', np.mean(known_source_target)
    #print '**********************************'
    #print '% triples (relation UNK): ', (global_stat['UNK-R'] * 100.0 / global_stat['all']), global_stat['all']
    #print '**********************************'


def get_train_valid_stream(KB):
    # training stream creation...
    train_stream = []
    for dict_key in KB.train_data.keys():
        if dict_key.endswith('-R'):
            train_stream.extend(KB.train_data[dict_key])
    # create random cronological ordering ....
    for k in range(5):
        random.shuffle(train_stream)

    # validation stream creation...
    valid_stream = []
    for dict_key in KB.valid_data.keys():
        if dict_key.endswith('-R'):
            valid_stream.extend(KB.valid_data[dict_key])
    # create random cronological ordering ....
    for k in range(5):
        random.shuffle(valid_stream)

    return train_stream, valid_stream


def get_neg_sample_dom(s_q_instance, t_q_instance, r_rel, KB, dom_flag=False):
    s_answer_set = s_q_instance['ans_set']
    t_answer_set = t_q_instance['ans_set']

    if dom_flag:
        cand_s_set = []
        for s_dom in KB.rel_domains[r_rel]['s_dom']:
            cand_s_set.extend(KB.domain_entity_set[s_dom])
        cand_s_set = set(cand_s_set)

        cand_t_set = []
        for t_dom in KB.rel_domains[r_rel]['t_dom']:
            cand_t_set.extend(KB.domain_entity_set[t_dom])
        cand_t_set = set(cand_t_set)
    else:
        cand_s_set = set(KB.entity_vocab.keys())
        cand_t_set = set(KB.entity_vocab.keys())

    # get negative set and sample
    s_neg_sampling_set = cand_s_set.difference(s_answer_set)
    # get negative set and sample
    t_neg_sampling_set = cand_t_set.difference(t_answer_set)

    # get neg sets ...
    neg_s_full = [KB.entity_vocab.get(s_neg) for s_neg in s_neg_sampling_set]
    neg_t_full = [KB.entity_vocab.get(t_neg) for t_neg in t_neg_sampling_set]

    pos_s_full = set([KB.entity_vocab.get(s_pos) for s_pos in s_answer_set])
    pos_t_full = set([KB.entity_vocab.get(t_pos) for t_pos in t_answer_set])

    return neg_s_full, neg_t_full, cand_s_set, cand_t_set, pos_s_full, pos_t_full


# def get_neg_sample(data_instance, controller):
#     s_node = data_instance.sourceNode
#     r_rel = data_instance.rel
#     t_domain = controller.KB.rel_target_domains[r_rel]
#
#     # accumulate true triples'.. targets..
#     exclusion_set = {data_instance.answer}
#     for node, rel_list in controller.KB.node_index[s_node].items():
#         if r_rel in rel_list:
#             exclusion_set.add(node)
#
#     # get negative set and sample
#     sampling_set = t_domain.difference(exclusion_set)
#     neg_sample = random.sample(sampling_set, min(len(sampling_set), controller.neg_samples_per_triple))
#     neg_t = [controller.KB.entity_vocab.get(t_neg) for t_neg in neg_sample]
#     neg_t = pad_arr_seq(neg_t, controller.neg_samples_per_triple, 0)
#     return neg_t


def generate_pos_neg_sample(data_chunk, KB, data_store, i, out_q):
    data_sample_gen = []

    for triple in data_chunk:
        node1 = triple[0]
        rel = triple[1]
        node2 = triple[2]

        head_q_instance = data_store[node1 + '-ENT']["-#" + rel + "#" + node2]
        tail_q_instance = data_store[node1 + '-ENT'][node1 + "#" + rel + '#-']

        neg_s_full, neg_t_full, cand_s_set, cand_t_set, s_answer_set, t_answer_set\
                             = get_neg_sample_dom(head_q_instance, tail_q_instance, rel, KB)
        data_tup = (
            [KB.entity_vocab.get(node1)],
            [KB.rel_vocab.get(rel)], [KB.entity_vocab.get(node2)],
            neg_s_full, neg_t_full, s_answer_set, t_answer_set)  #cand_s_set, cand_t_set, s_answer_set, t_answer_set)

        data_sample_gen.append((triple, data_tup))
    out_q.put(data_sample_gen)


def get_full_batch_dataset(triple_data_stream, KB, mode, n_procs=10):
    data_X = []

    if mode == 'T':
        data_store = KB.train_data
    else:
        data_store = KB.valid_data

    nprocs = n_procs
    out_q = Queue()
    chunksize = int(math.ceil(len(triple_data_stream) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        print ('process-' + str(i) + 'created')
        # sys.stderr.write("\r")
        # sys.stderr.write('process-'+str(i)+'created')
        # sys.stderr.flush()
        p = Process(target=generate_pos_neg_sample,
            args=(triple_data_stream[chunksize * i:chunksize * (i + 1)], KB, data_store, i, out_q))
        procs.append(p)
        p.start()

    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    for i in range(nprocs):
        data_X.extend(out_q.get())
        print (' ---> process-', i, 'output')

        # Wait for all worker processes to finish
    for p in procs:
        p.join()
        # print 'data_X prepared', len(data_X)
    return data_X


def load_or_initialize_buffers(KB_name):
    if os.path.isfile('./KB_dumps/' + KB_name + 'perf_buff.pickle'):
        with open('./KB_dumps/' + KB_name + 'perf_buff.pickle', "rb") as input_file:
            performance_buff = pickle.load(input_file)
    else:
        performance_buff = {'rel': {}, 'ent': {}}

    if os.path.isfile('./KB_dumps/' + KB_name + 'thresh_buff.pickle'):
        with open('./KB_dumps/' + KB_name + 'thresh_buff.pickle', "rb") as input_file:
            thresh_buff = pickle.load(input_file)
    else:
        thresh_buff = {'rel': {}, 'ent': {}}

    return performance_buff, thresh_buff


def save_perf_thresh_buffers(KB_name, perf_buff, thresh_buff):
    with open('./KB_dumps/' + KB_name + 'perf_buff.pickle', 'w') as out_file:
        pickle2.dump(perf_buff, out_file)
    with open('./KB_dumps/' + KB_name + 'thresh_buff.pickle', 'w') as out_file:
        pickle2.dump(thresh_buff, out_file)


