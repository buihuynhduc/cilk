from util_helper import *
random.seed(1000)


def generate_pos_neg_sample_online(data_chunk, KB, data_store):
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
            neg_s_full, neg_t_full, s_answer_set, t_answer_set)   #cand_s_set, cand_t_set

        data_sample_gen.append((triple, data_tup))
    return data_sample_gen


def get_sampled_dataset(input_data, q_ent, q_rel, training_mode, KB, sample_size, mode):
    sampled_dataset = []

    if training_mode == 'random':
        all_instances = []
        for dict_key in input_data.keys():
            if dict_key.endswith('-R'):
                all_instances.extend(input_data[dict_key])
        sampled_dataset = random.sample(all_instances, min(len(all_instances), sample_size))

    elif training_mode == 'rel':
        if q_rel + '-R' in input_data:
            sampled_dataset = random.sample(input_data[q_rel+'-R'], min(len(input_data[q_rel+'-R']), sample_size))

    elif training_mode == 'ent':
        if q_ent + '-NDX' in input_data:
            sampled_dataset = random.sample(input_data[q_ent+'-NDX'], min(len(input_data[q_ent+'-NDX']), sample_size))

    elif training_mode == 'both':
        sampled_ent_triples = []
        if q_ent+'-NDX' in input_data:
             sampled_ent_triples = random.sample(input_data[q_ent+'-NDX'], min(len(input_data[q_ent+'-NDX']), int(sample_size/2.0)))

        if q_rel+'-R' in input_data:
             filtered_rel_triples = input_data[q_rel+'-R'].difference(set(sampled_ent_triples))
             sampled_trip = random.sample(filtered_rel_triples, min(len(filtered_rel_triples), (sample_size - len(sampled_ent_triples))))
             sampled_dataset.extend(sampled_trip)

    sample_batch_data = generate_pos_neg_sample_online(sampled_dataset, KB, input_data)

    data_vecs = []
    dataset = []
    for data_instance, data_tup in sample_batch_data:
        data_vecs.append(data_tup)
        dataset.append(data_instance)

    return dataset, data_vecs


def get_sample_mean(data_pred, data_ans_set):
    true_label = np.zeros_like(data_pred)
    for ent_id in data_ans_set:
        true_label[ent_id] = 1.0
    pred_score = data_pred

    pos_dist = true_label * pred_score
    pos_dist = pos_dist[pos_dist > 0.0]

    neg_dist = (1 - true_label) * pred_score
    neg_dist = neg_dist[neg_dist > 0.0]

    sample_mean = (np.mean(pos_dist) + np.mean(neg_dist)) / 2.0
    # sample_mean = (np.min(pos_dist) + np.max(neg_dist)) / 2.0
    return sample_mean


def compute_rel_and_ent_thresholds(s_ent, q_rel, t_ent, data_pred_s, data_s_ans,
                                   data_pred_t, data_t_ans, per_sample_thresh):
    t_ent_mean = get_sample_mean(data_pred_s, data_s_ans)
    s_ent_mean = get_sample_mean(data_pred_t, data_t_ans)

    if s_ent in per_sample_thresh['ent']:
        per_sample_thresh['ent'][s_ent].append(s_ent_mean)
    else:
        per_sample_thresh['ent'][s_ent] = [s_ent_mean]

    if t_ent in per_sample_thresh['ent']:
        per_sample_thresh['ent'][t_ent].append(t_ent_mean)
    else:
        per_sample_thresh['ent'][t_ent] = [t_ent_mean]

    if q_rel in per_sample_thresh['rel']:
        per_sample_thresh['rel'][q_rel].append((s_ent_mean + t_ent_mean) / 2.0)
    else:
        per_sample_thresh['rel'][q_rel] = [(s_ent_mean + t_ent_mean) / 2.0]


def get_pred_threshold(q_ent, q_rel, agent):
    if q_ent in agent.thresh_buff['ent']:
        ent_thresh = agent.thresh_buff['ent'][q_ent]
    else:
        ent_thresh = 0.0

    if q_rel in agent.thresh_buff['rel']:
        rel_thresh = agent.thresh_buff['rel'][q_rel]
    else:
        rel_thresh = 0.0

    if agent.controller.threshold_type == 'both':
        pred_thresh = max(ent_thresh, rel_thresh) # satisfy both threshold ..
    elif agent.controller.threshold_type == 'ent':
        pred_thresh = ent_thresh
    elif agent.controller.threshold_type == 'rel':
        pred_thresh = rel_thresh
    elif agent.controller.threshold_type == 'any':
        pred_thresh = min(ent_thresh, rel_thresh)  # satisfy either threshold ..
    else:
        pred_thresh = -1000.0

    #print agent.controller.threshold_type

    #print 'thresholds: ', ent_thresh, rel_thresh, pred_thresh
    return pred_thresh