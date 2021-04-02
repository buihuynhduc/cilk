import pickle as pickle2
import numpy as np, os


def update_MR_and_MRR(q_rel, RR, ent_rank, MRR_rel, MRank_rel):
    if RR != -1.0 and ent_rank != -1.0:
        if q_rel in MRR_rel:
            MRR_rel[q_rel].append(RR)
            MRank_rel[q_rel].append(ent_rank)
        else:
            MRR_rel[q_rel] = [RR]
            MRank_rel[q_rel] = [ent_rank]


def update_hits_k(q_rel, hits_k_res_rel, hits_k_res):
    if q_rel in hits_k_res_rel:
        for top_k, val in hits_k_res.items():
            hits_k_res_rel[q_rel][top_k].append(val)
    else:
        hits_k_res_rel[q_rel] = {}
        for top_k, val in hits_k_res.items():
            hits_k_res_rel[q_rel][top_k] = [val]


def get_avg_MRR(MRR_rel_s, MRR_rel_t):
    MRR_rel_s_all = []
    MRR_rel_t_all = []

    for relq in MRR_rel_s:
        MRR_rel_s_all.extend(MRR_rel_s[relq])

    for relq in MRR_rel_t:
        MRR_rel_t_all.extend(MRR_rel_t[relq])

    # print len(MRR_rel_s_all) > 0
    # print len(MRR_rel_t_all) > 0

    # avg_MRR_s = np.mean([np.mean(MRR_rel_s[relq]) for relq in MRR_rel_s])
    # avg_MRR_t = np.mean([np.mean(MRR_rel_t[relq]) for relq in MRR_rel_t])
    avg_MRR = (np.nanmean(MRR_rel_s_all) + np.nanmean(MRR_rel_t_all)) / 2.0
    return avg_MRR

#
# def get_avg_MRank(MRank_rel_s, MRank_rel_t):
#
#     avg_MR_s = np.mean([np.mean(MRank_rel_s['k'][relq]) for relq in MRank_rel_s['k']])
#     avg_MR_t = np.mean([np.mean(MRank_rel_t['k'][relq]) for relq in MRank_rel_t['k']])
#     avg_MR = (avg_MR_s + avg_MR_t) / 2.0
#     return avg_MR


def get_hits_at_k(hits_k, hits_k_res_rel_s, hits_k_res_rel_t):
    s_avg_hits_k = {}
    t_avg_hits_k = {}

    for top_k in hits_k:
        s_hits_k_all = []
        for relq in hits_k_res_rel_s:
            s_hits_k_all.extend(hits_k_res_rel_s[relq][top_k])

        s_avg_hits_k[top_k] = 100.0 * np.mean(s_hits_k_all)

    for top_k in hits_k:
        t_hits_k_all = []
        for relq in hits_k_res_rel_t:
            t_hits_k_all.extend(hits_k_res_rel_t[relq][top_k])

        t_avg_hits_k[top_k] = 100.0 * np.mean(t_hits_k_all)

    avg_hits_k = {}
    for top_k in hits_k:
        avg_hits_k[top_k] = (s_avg_hits_k[top_k] + t_avg_hits_k[top_k]) / 2.0

    return avg_hits_k


def evaluate_test_data(eval_buff, eval_ckpt, known_info_map, ent_flag, rel_flag):
    hits_k = [1, 3, 10]
    MRank_rel_s = {}
    MRank_rel_t = {}
    MRR_rel_s = {}
    MRR_rel_t = {}
    hits_k_res_rel_s = {}
    hits_k_res_rel_t = {}

    rejection_dict = {'0-0': 0, '0-1':0, '1-0':0, '1-1':0}

    num_count = 0
    for query_tup in eval_buff[:eval_ckpt]:
        query, RR, hits_k_res, ent_rank, ans_exists, ow_k_res = query_tup

        q_ent = query.split("|")[0]
        q_rel = query.split("|")[1]
        q_type = query.split("|")[2]

        rejection_dict[str(int(ans_exists)) +'-'+ str(int(ow_k_res[1]))] += 1

        # if not ans_exists:
        #      continue
        #
        # if not ow_k_res[1]:
        #     continue

        if ent_flag == '*' and rel_flag == '*':
            if q_ent in known_info_map['ent'] and q_rel in known_info_map['rel']:
                continue

        if ent_flag == 'k':
            if q_ent not in known_info_map['ent']:
                continue
        elif ent_flag == 'u':
            if q_ent in known_info_map['ent']:
                continue

        if rel_flag == 'k':
            if q_rel not in known_info_map['rel']:
                continue
        elif rel_flag == 'u':
            if q_rel in known_info_map['rel']:
                continue

        num_count += 1
        if q_type == 't':
             update_MR_and_MRR(q_rel, RR, ent_rank, MRR_rel_t, MRank_rel_t)
             update_hits_k(q_rel, hits_k_res_rel_t, hits_k_res)
        else:
             update_MR_and_MRR(q_rel, RR, ent_rank, MRR_rel_s, MRank_rel_s)
             update_hits_k(q_rel, hits_k_res_rel_s, hits_k_res)

    print ('Count:',  num_count)
    avg_MRR= get_avg_MRR(MRR_rel_s, MRR_rel_t)
    #avg_MR = get_avg_MRR(MRank_rel_s, MRank_rel_t)
    avg_hits_k = get_hits_at_k(hits_k, hits_k_res_rel_s, hits_k_res_rel_t)

    return avg_MRR, avg_hits_k, rejection_dict


if __name__ == "__main__" :
    #KB_name = 'Wordnet_'
    KB_name = 'Nell_'
    result_folder = 'bth_btr_c1_ef2'

    eval_modes = [('k', 'k'), ('k', 'u'), ('u', 'k'), ('u', 'u'), ('-', '-'), ('*', '*')]
    #eval_modes = [('k', 'k'), ('-', 'u'), ('u', '-'), ('-', '-'), ('*', '*')]
    #rel_flag = 'u'
    #ent_flag = 'k'

    print (os.getcwd())

    with open('./KB_dumps/NELL_140_250D/'+ result_folder+'/'+ KB_name +'eval_buff.pickle', "rb") as input_file:
         evaluation_buffer = pickle2.load(input_file)
    with open('./KB_dumps/NELL_140_250D/'+ result_folder+'/' +KB_name + 'known_info_map.pickle', "rb") as input_file:
        known_info_map = pickle2.load(input_file)
    # metric stores ...
    test_stream_len = len(evaluation_buffer)
    eval_checkpoints = [0.5, 1.0]
    eval_index = [int(ckpt * test_stream_len) for ckpt in eval_checkpoints]

    for eval_mode in eval_modes:
        rel_flag = eval_mode[0]
        ent_flag = eval_mode[1]

        print ('\n ######### EVAL TYPE:   Rel = ', rel_flag, " Ent = ", ent_flag, " ############\n")

        for eval_id in eval_index:
            avg_MRR, avg_hits_k, reject_dict = evaluate_test_data(evaluation_buffer, eval_id, known_info_map, ent_flag, rel_flag)

            print ('\n == TEST DATA OBSERVED: ', round((100.0 * eval_id) / test_stream_len, 2), " ====\n")
            print ('test_MRR: ', round(avg_MRR, 4), '\n', \
                'test_hits@1: ', round(avg_hits_k[1], 4), \
                'test_hits@3: ', round(avg_hits_k[3], 4), \
                'test_hits@10: ', round(avg_hits_k[10], 4), ' \n')

            print ('reject_dict: ', reject_dict)
            pred = (reject_dict['1-1']*1.0) / (reject_dict['1-1'] + reject_dict['1-0'])
            reject = (reject_dict['0-0'] * 1.0) / (reject_dict['0-0'] + reject_dict['0-1'])

            print ('recall of prediction: ', pred)
            print ('recall of rejection: ', reject)

# 'test_hits@3: ', round(avg_hits_k[3], 4), \