from online_training_helper import *
from batch_training_helper import *


def train_and_evaluate_inf_model(q_ent, q_rel, distmult_model, tr_batch_size, agent, max_epoch, lr):
    hits_k = [1, 3, 10]

    train_dataset, train_vecs = get_sampled_dataset(agent.KB.train_data, q_ent, q_rel,
                                                      agent.controller.training_mode, agent.KB, agent.controller.sample_size, 'T')
    valid_dataset, valid_vecs = get_sampled_dataset(agent.KB.valid_data, q_ent, q_rel,
                                                      agent.controller.training_mode, agent.KB, agent.controller.sample_size/10, 'V')
    cand_ent_size = len(agent.KB.entity_vocab)

    if len(train_vecs) > 0:
        num_batches = int(((1.0*len(train_vecs))/tr_batch_size)+1)
        #print 'num_batches ..', num_batches

        # train_s, train_r, train_t, train_s_neg, train_t_neg, train_s_ans, train_t_ans \
        #     = get_next_data_batch(train_vecs, 0, len(train_vecs), agent.controller.neg_samples_per_triple)

        # start training ...
        #start_time = time.time()
        for i in xrange(max_epoch):

            # random shuffling of train data
            train_dataset, train_vecs = get_shuffled_train_data(train_dataset, train_vecs)

            # mini-batch training ...
            #avg_loss = 0
            for j in xrange(num_batches):
                tr_batch_s, tr_batch_r, tr_batch_t, tr_batch_s_neg, tr_batch_t_neg, tr_batch_s_ans, tr_batch_t_ans = \
                              get_next_data_batch(train_vecs, j, tr_batch_size, agent.controller.neg_samples_per_triple)

                _, loss = distmult_model.train(tr_batch_s, tr_batch_r, tr_batch_t, tr_batch_s_neg, tr_batch_t_neg, lr)
                #avg_loss += loss
                sys.stderr.write("\r")
                sys.stderr.write("Epoch- %d processing %0.3f" % ((i+1), round(j*100.0/num_batches, 3)))
                sys.stderr.flush()

        if len(valid_vecs) > 0:
            # get validation data ...
            valid_s, valid_r, valid_t, valid_s_neg, valid_t_neg, valid_s_ans, valid_t_ans \
                = get_next_data_batch(valid_vecs, 0, len(valid_vecs), agent.controller.neg_samples_per_triple)

            valid_loss, valid_pred_s, valid_pred_t = distmult_model.evaluate(valid_s, valid_r, valid_t,
                                                                             valid_s_neg, valid_t_neg, cand_ent_size)
            avg_MRR_vd, avg_hits_k_vd, avg_MR_vd = \
                    evaluate_and_record_performance(valid_s_ans, valid_t_ans, valid_pred_s, valid_pred_t,
                                                 agent, valid_dataset, False, hits_k, 'valid')


def run_inference(test_q, distmult_model, agent):
    '''
    'q_ent': node2, 'q_rel': rel, 'ans_set': {node1},
    'a_type': 'h', 'mode': mode, 'interaction_limit': interaction_lt
    '''
    hits_k = [1, 3, 10]

    # prepare test input ....
    q_ent_vec = np.array([[agent.KB.entity_vocab.get(test_q['q_ent'])]])
    q_rel_vec = np.array([[agent.KB.rel_vocab.get(test_q['q_rel'])]])
    q_ans_set = set([agent.KB.entity_vocab.get(ent) for ent in test_q['ans_set']])
    cand_ent_size = len(agent.KB.entity_vocab)

    #print '.... Evaluating on Test Data ...'
    test_key = test_q['q_ent'] + "|" + test_q['q_rel'] + "|" + test_q['a_type']
    print( test_key )
    print (q_ent_vec, q_rel_vec, cand_ent_size)
    if test_q['a_type'] == 'h':
        pred_score = distmult_model.predict_source(q_rel_vec, q_ent_vec, cand_ent_size)
    else:
        pred_score = distmult_model.predict_target(q_ent_vec, q_rel_vec, cand_ent_size)

    pred_thresh = get_pred_threshold(test_q['q_ent'], test_q['q_rel'], agent)
    #print '===', pred_score[0]
    RR, hits_k_res, ent_Rank, ans_exists, ow_k_res = evaluate_query(pred_score[0], q_ans_set, pred_thresh, hits_k, True)

    agent.evaluation_buffer.append((test_key, RR, hits_k_res, ent_Rank, ans_exists, ow_k_res))