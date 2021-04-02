import math, time, sys, os
from util_helper\
import get_processing_time, pad_arr_seq
import numpy as np, random
from online_training_helper import compute_rel_and_ent_thresholds
from operator import itemgetter
random.seed(1000)


def get_next_data_batch(dataset, itr, batch_size, neg_sample_size):
    batch = dataset[(itr * batch_size):(itr * batch_size) + batch_size]

    batch_s = np.array([_[0] for _ in batch])
    batch_r = np.array([_[1] for _ in batch])
    batch_t = np.array([_[2] for _ in batch])
    #batch_t = np.array([random.choice(_[3]) for _ in batch])

    batch_s_neg = np.array([pad_arr_seq(random.sample(_[3], min(len(_[3]), neg_sample_size)), neg_sample_size, 0) for _ in batch])
    batch_t_neg = np.array([pad_arr_seq(random.sample(_[4], min(len(_[4]), neg_sample_size)), neg_sample_size, 0) for _ in batch])

    batch_s_ans = [_[5] for _ in batch]
    batch_t_ans = [_[6] for _ in batch]

    return np.reshape(batch_s, (len(batch_s), 1)), np.reshape(batch_r, (len(batch_r), 1)), \
           np.reshape(batch_t, (len(batch_t), 1)), np.reshape(batch_s_neg, (len(batch_s_neg), neg_sample_size)), \
           np.reshape(batch_t_neg, (len(batch_t_neg), neg_sample_size)), batch_s_ans, batch_t_ans


def get_shuffled_train_data(data_set, data_sample):
    zipped_data = zip(data_set, data_sample)

    for k in range(5):
        random.shuffle(zipped_data)

    data_x, data_y = zip(*zipped_data)
    return list(data_x), list(data_y)


def train_and_evaluate_inf_model_full_batch(trainset, validset, distmult_model, tr_batch_size, agent, max_epoch):
    hits_k = [1, 3, 10]
    if agent.controller.init_train_epoch > 10:
        eval_interval = (agent.controller.init_train_epoch-2)
    else:
        eval_interval = 100

    print( 'starting training ...')
    test_ent_size = len(agent.KB.entity_vocab)
    print( '# train_sample / # valid sample: ', len(trainset), len(validset))

    train_vecs = []
    train_dataset = []
    for data_instance, data_tup in trainset:
        train_vecs.append(data_tup)
        train_dataset.append(data_instance)

    valid_vecs = []
    valid_dataset = []
    for data_instance, data_tup in validset:
        valid_vecs.append(data_tup)
        valid_dataset.append(data_instance)

    # start training ...
    if len(train_vecs) > 0 and len(valid_vecs) > 0:
        num_batches = int(((1.0 * len(train_vecs))/tr_batch_size)+1)
        print ('num_batches ..', num_batches)

        # start epoch ...
        start_time = time.time()
        for i in range(max_epoch):

            # random shuffling of train data
            train_dataset, train_vecs = get_shuffled_train_data(train_dataset, train_vecs)

            # mini-batch training ...
            avg_loss = 0.0
            avg_tr_MR = 0.0
            avg_tr_MRR = 0.0
            avg_tr_hits = {}

            lr = agent.controller.lr1
            if i > 80:
                lr = agent.controller.lr2

            for k in hits_k:
              avg_tr_hits[k] = 0.0

            for j in range(num_batches):
                train_dataset_batch = train_dataset[(j * tr_batch_size):(j * tr_batch_size) + tr_batch_size]
                tr_batch_s, tr_batch_r, tr_batch_t, tr_batch_s_neg, tr_batch_t_neg, tr_batch_s_ans, tr_batch_t_ans = \
                                           get_next_data_batch(train_vecs, j, tr_batch_size, agent.controller.neg_samples_per_triple)

                #print np.shape(tr_batch_s), np.shape(tr_batch_r), np.shape(tr_batch_t), np.shape(tr_batch_s_neg)

                _, loss = distmult_model.train(tr_batch_s, tr_batch_r, tr_batch_t, tr_batch_s_neg, tr_batch_t_neg, lr)
                avg_loss += loss
                sys.stderr.write("\r")
                sys.stderr.write("Epoch- %d processing %0.3f" % ((i+1), round(j*100.0/num_batches, 3)))
                sys.stderr.flush()

                # if i > 0 and i % eval_interval == 0:
                #     loss, train_pred_s, train_pred_t = distmult_model.evaluate(tr_batch_s, tr_batch_r, tr_batch_t,
                #                                                tr_batch_s_neg, tr_batch_t_neg, test_ent_size)
                #     avg_MRR, avg_hits_k, avg_MR = \
                #         evaluate_and_record_performance(tr_batch_s_ans, tr_batch_t_ans,
                #                         train_pred_s, train_pred_t, agent, train_dataset_batch, False, hits_k, 'train')
                #
                #     avg_tr_MRR += avg_MRR
                #     avg_tr_MR += avg_MR
                #     for k in hits_k:
                #        avg_tr_hits[k] += avg_hits_k[k]

            # if i > 0 and i % eval_interval == 0:
            #     avg_tr_MRR /= num_batches
            #     avg_tr_MR /= num_batches
            #     for k in hits_k:
            #         avg_tr_hits[k] /= num_batches

            analyze = False
            if i > 0 and i % eval_interval == 0:
                analyze = True

            # Evaluate on Validation ..........
            avg_vd_loss = 0.0
            avg_vd_MRR = 0.0
            avg_vd_MR = 0.0
            avg_vd_hits = {}
            for k in hits_k:
               avg_vd_hits[k] = 0.0

            num_vd_batches = int(((1.0 * len(valid_vecs)) / tr_batch_size) + 1)

            for j in range(num_vd_batches):
                vd_dataset_batch = valid_dataset[(j * tr_batch_size):(j * tr_batch_size) + tr_batch_size]
                vd_batch_s, vd_batch_r, vd_batch_t, vd_batch_s_neg, vd_batch_t_neg, vd_batch_s_ans, vd_batch_t_ans \
                    = get_next_data_batch(valid_vecs, j, tr_batch_size,  agent.controller.neg_samples_per_triple)

                valid_loss, valid_pred_s, valid_pred_t = distmult_model.evaluate(vd_batch_s, vd_batch_r, vd_batch_t,
                                                                 vd_batch_s_neg, vd_batch_t_neg, test_ent_size)

                avg_vd_loss += valid_loss

                if i > 0 and i % eval_interval == 0:
                    avg_MRR, avg_hits_k, avg_MR = \
                        evaluate_and_record_performance(vd_batch_s_ans, vd_batch_t_ans, valid_pred_s, valid_pred_t,
                                                        agent, vd_dataset_batch, analyze, hits_k, 'valid')

                    avg_vd_MRR += avg_MRR
                    avg_vd_MR += avg_MR
                    for k in hits_k:
                        avg_vd_hits[k] += avg_hits_k[k]

            avg_vd_loss /= num_vd_batches
            if i > 0 and i % eval_interval == 0:
                avg_vd_MRR /= num_vd_batches
                avg_vd_MR /= num_vd_batches
                for k in hits_k:
                    avg_vd_hits[k] /= num_vd_batches

            print ('-- Tr_loss: ', round(avg_loss/num_batches, 5), '-- Vd_loss: ', round(avg_vd_loss, 5), \
                                '---- time: ', get_processing_time(start_time))

            if i > 0 and i % eval_interval == 0:
                # print 'tr_MR: ', round(avg_tr_MR, 4), 'tr_MRR: ', round(avg_tr_MRR, 4), \
                #     'tr_hits@1: ', round(avg_tr_hits[1], 4), 'tr_hits@10: ', round(avg_tr_hits[10], 4)
                print (' vd_MR: : ', round(avg_vd_MR, 4), ' vd_MRR: : ', round(avg_vd_MRR, 4), \
                    'vd_hits@1: ', round(avg_vd_hits[1], 4), 'vd_hits@10: ', round(avg_vd_hits[10], 4))

        print( '.... Training Complete .....')


def evaluate_query(q_pred, true_answer, pred_thresh, hits_k, is_analyze):
        pred_list = zip(np.arange(len(q_pred)), q_pred)
        answer_sorted = sorted(pred_list, key=itemgetter(1), reverse=True)  # highest ranked entity

        hits_k_res = {}
        ow_k_res = {}
        RR = -1.0
        ent_Rank = -1.0

        ans_exists = True
        predicted = False

        # reciprocal rank ...
        for rank, value in enumerate(answer_sorted, 1):
            ent_id = value[0]

            if ent_id in true_answer:
               RR = (1.0 / rank)
               ent_Rank = rank
               break

        if RR == -1.0 and ent_Rank == -1.0:     # if not found, only hits is updated ...# RR does not make sense as entity is not found..
           #print '----', answer_sorted
           max_pred_score = answer_sorted[0][1]
           ans_exists = False

           if max_pred_score > pred_thresh:
               predicted = True
               for top_k in hits_k:
                   hits_k_res[top_k] = 0.0
                   ow_k_res[top_k] = predicted
           else:
               predicted = False
               for top_k in hits_k:
                   hits_k_res[top_k] = 1.0
                   ow_k_res[top_k] = predicted

           return RR, hits_k_res, ent_Rank, ans_exists, ow_k_res

        # entity is present
        for top_k in hits_k:
            top_k_answers = []
            for ent_tup in answer_sorted[:top_k]:
                if ent_tup[1] > pred_thresh:
                     top_k_answers.append(ent_tup)
            pred_answers = {ent_id for ent_id, _ in top_k_answers}

            if len(pred_answers) > 0:
                predicted = True
            else:
                predicted = False

            ow_k_res[top_k] = predicted
            # if is_analyze and top_k == hits_k[-1]:
            #     write_pred_results(q_ent, q_rel, true_answer, top_k_answers, agent)

            if len(true_answer.intersection(pred_answers)) > 0:
                hits_k_res[top_k] = 1.0
            else:
                hits_k_res[top_k] = 0.0

        return RR, hits_k_res, ent_Rank, ans_exists, ow_k_res


def evaluate_and_record_performance(data_s_ans, data_t_ans, data_pred_s, data_pred_t, agent, dataset, is_analyze, hits_k, mode):

    MRank_rel_s = {}
    MRank_rel_t = {}
    MRR_rel_s = {}
    MRR_rel_t = {}
    MRR_ent = {}
    hits_k_res_rel_s = {}
    hits_k_res_rel_t = {}

    per_sample_thresh = {'rel':{}, 'ent':{}}

    for i in range(0, len(dataset), 1):
        curr_s_ent = dataset[i][0]
        curr_q_rel = dataset[i][1]
        curr_t_ent = dataset[i][2]
        pred_thresh = 0.0

        t_RR, t_hits_k_res, t_ent_rank, ans_exists, ow_k_res = evaluate_query(data_pred_t[i], data_t_ans[i],
                                                          pred_thresh, hits_k, is_analyze)
        s_RR, s_hits_k_res, s_ent_rank, ans_exists, ow_k_res = evaluate_query(data_pred_s[i], data_s_ans[i],
                                                          pred_thresh, hits_k, is_analyze)

        update_MR_and_MRR(curr_q_rel, t_RR, t_ent_rank, MRR_rel_t, MRank_rel_t, MRR_ent, curr_s_ent)
        update_MR_and_MRR(curr_q_rel, s_RR, s_ent_rank, MRR_rel_s, MRank_rel_s, MRR_ent, curr_t_ent)

        update_hits_k(curr_q_rel, hits_k_res_rel_t, t_hits_k_res)
        update_hits_k(curr_q_rel, hits_k_res_rel_s, s_hits_k_res)

        if mode == 'valid':
            compute_rel_and_ent_thresholds(curr_s_ent, curr_q_rel, curr_t_ent,
                                            data_pred_s[i], data_s_ans[i], data_pred_t[i], data_t_ans[i],
                                            per_sample_thresh)

    if mode == 'valid':
        for ent in per_sample_thresh['ent'].keys():
            agent.thresh_buff['ent'][ent] = np.mean(per_sample_thresh['ent'][ent])

        for reln in per_sample_thresh['rel'].keys():
            agent.thresh_buff['rel'][reln] = np.mean(per_sample_thresh['rel'][reln])

        for q_ent in MRR_ent:
            agent.performance_buff['ent'][q_ent] = np.mean(MRR_ent[q_ent])

    if len(MRR_rel_s) > 0:
        temp_s = {relq : np.mean(MRR_rel_s[relq]) for relq in MRR_rel_s}
        temp_t = {relq : np.mean(MRR_rel_t[relq]) for relq in MRR_rel_t}

        temp_MRR_rel = {}
        for relq in temp_s:
            temp_MRR_rel[relq] = (np.mean(temp_s[relq]) + np.mean(temp_t[relq])) / 2.0

            if mode == 'valid':
                agent.performance_buff['rel'][relq] = temp_MRR_rel[relq]
        avg_MRR = np.mean(temp_MRR_rel.values())
    else:
        avg_MRR = -1.0

    if len(MRank_rel_s) > 0:
        avg_MR_s = np.mean([np.mean(MRank_rel_s[relq]) for relq in MRank_rel_s])
        avg_MR_t = np.mean([np.mean(MRank_rel_t[relq]) for relq in MRank_rel_t])
        avg_MR = (avg_MR_s + avg_MR_t) / 2.0
    else:
        avg_MR = -1.0

    s_avg_hits_k = {}
    for top_k in hits_k:
        s_avg_hits_k[top_k] = 100.0 * np.mean([np.mean(hits_k_res_rel_s[relq][top_k]) for relq in hits_k_res_rel_s])

    t_avg_hits_k = {}
    for top_k in hits_k:
        t_avg_hits_k[top_k] = 100.0 * np.mean([np.mean(hits_k_res_rel_t[relq][top_k]) for relq in hits_k_res_rel_t])

    avg_hits_k = {}
    for top_k in hits_k:
        avg_hits_k[top_k] = (s_avg_hits_k[top_k] + t_avg_hits_k[top_k]) / 2.0

    return avg_MRR, avg_hits_k, avg_MR


def update_MR_and_MRR(q_rel, RR, ent_rank, MRR_rel, MRank_rel, MRR_ent, q_ent):
    if RR != -1.0 and ent_rank != -1.0:
        if q_rel in MRR_rel:
            MRR_rel[q_rel].append(RR)
            MRank_rel[q_rel].append(ent_rank)
        else:
            MRR_rel[q_rel] = [RR]
            MRank_rel[q_rel] = [ent_rank]

        if q_ent in MRR_ent:
            MRR_ent[q_ent].append(RR)
        else:
            MRR_ent[q_ent] = [RR]


def update_hits_k(q_rel, hits_k_res_rel, hits_k_res):
    if q_rel in hits_k_res_rel:
        for top_k, val in hits_k_res.items():
            hits_k_res_rel[q_rel][top_k].append(val)
    else:
        hits_k_res_rel[q_rel] = {}
        for top_k, val in hits_k_res.items():
            hits_k_res_rel[q_rel][top_k] = [val]


def write_pred_results(s_ent, q_rel, t_ent_set, ranklist, agent):
    base_path = './'+agent.controller.KB_name+'_result_dict'
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    file = open(base_path+ '/q_rel_res.txt', 'a')
    file.write("Q: ( "+s_ent+" , "+q_rel+ " , "+str(t_ent_set)+" )\nRankL: [")
    for ent_id, pred_score in ranklist:
        file.write(agent.KB.inv_entity_vocab[ent_id]+'-->'+str(round(pred_score, 5))+' ; ')
    file.write(']\n')
    file.close()
    #print 'query log written...'

