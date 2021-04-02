import tensorflow as tf


class DistMult(object):

    def __init__(self, sess, rel_size, ent_size, max_neg, embed_dim, lamda):
        self.sess=sess
        self.max_num_rels = rel_size
        self.max_num_entities = ent_size
        self.embedding_dim=embed_dim
        self.max_neg_sample = max_neg
        self.lamda=lamda
        self.use_pre_train = False

        self.network_params = tf.compat.v1.trainable_variables()
        self.train_layer()
        self.prediction_layer()

    def train_layer(self):
        tf.compat.v1.disable_eager_execution()
        self.s_ent_neg = tf.compat.v1.placeholder(shape=[None, self.max_neg_sample], dtype=tf.int32, name='cand_source_ents')
        self.t_ent_neg = tf.compat.v1.placeholder(shape=[None, self.max_neg_sample], dtype=tf.int32, name='cand_tail_ents')

        # get pos triple predictions ...
        pred_score_pos = tf.compat.v1.tile(self.pos_score, [1, self.max_neg_sample])  # None x 7

        # reshaping and tiling neg_s prediction ..
        s_neg_ent_emb = tf.nn.embedding_lookup(self.entity_embedding, self.s_ent_neg)  # say, None x 7 x 3
        pred_score_s_neg = tf.multiply(s_neg_ent_emb, self.r_rel_emb)   # None x 7 x 3 *  None x 1 x 3 ---> None x 7 x 3
        pred_score_s_neg = tf.matmul(pred_score_s_neg, self.t_ent_emb, transpose_b=True)  # None x 7  x 3 *  None x 1 x 3 ---> None x 7 x 1
        s_pred_score_neg = tf.reshape(pred_score_s_neg,
                                    [tf.shape(pred_score_s_neg)[0], tf.shape(pred_score_s_neg)[1]])  # None x 7

        # reshaping and tiling neg_t prediction ..
        t_neg_ent_emb = tf.nn.embedding_lookup(self.entity_embedding, self.t_ent_neg)     # say, None x 7 x 3
        t_q_emb_vec = self.s_ent_emb * self.r_rel_emb     # None x 1 x 3
        t_pred_score_neg = tf.matmul(t_q_emb_vec, t_neg_ent_emb, transpose_b=True)   # None x 1  x 3 *  None x 7  x 3 ---> None x 1 x 7
        t_pred_score_neg = tf.reshape(t_pred_score_neg, [tf.shape(t_pred_score_neg)[0], tf.shape(t_pred_score_neg)[2]])  # None x 7

        # loss computation
        s_net_score_mat = s_pred_score_neg - pred_score_pos + 1.0
        t_net_score_mat = t_pred_score_neg - pred_score_pos + 1.0
        s_per_instance_loss = tf.reduce_sum(tf.maximum(s_net_score_mat, tf.zeros_like(s_net_score_mat, dtype=tf.float32)), axis=1)
        t_per_instance_loss = tf.reduce_sum(tf.maximum(t_net_score_mat, tf.zeros_like(t_net_score_mat, dtype=tf.float32)), axis=1)
        self.loss = tf.reduce_sum(s_per_instance_loss) + tf.reduce_sum(t_per_instance_loss)

        # add L2-regularization ...
        self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(x) for x in self.network_params])
        self.loss = tf.reduce_mean(self.loss + 0.5 * self.lamda * self.reg_loss)

        # optimization...
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        #self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

    def entity_embedding_layer(self, is_trainable):
        with tf.compat.v1.variable_scope("entity_embedding", reuse=tf.compat.v1.AUTO_REUSE):
            embedding_encoder = tf.compat.v1.get_variable("entity_embedding", [self.max_num_entities, self.embedding_dim],
                                trainable=is_trainable, initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            if self.use_pre_train:
               self.embedding_placeholder = tf.compat.v1.placeholder(tf.float32, [self.max_num_entities, self.embedding_dim])
               self.embedding_init = embedding_encoder.assign(self.embedding_placeholder)
        return embedding_encoder

    def relation_embedding_layer(self, is_trainable):
        with tf.variable_scope("rel_embedding", reuse=tf.AUTO_REUSE):
            embedding_encoder = tf.get_variable("encoder_embedding", [self.max_num_rels, self.embedding_dim],
                                trainable=is_trainable, initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        return embedding_encoder

    def build_network(self):
        # categorical input
        tf.compat.v1.disable_eager_execution()
        self.s_ent = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.int16, name='pos_h')
        self.r_rel = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.int32, name='pos_r')
        self.t_ent = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.int32, name='pos_t')
        # mode
        self.train_mode = tf.compat.v1.placeholder(tf.bool, name='train_mode')

        # embedding ...
        self.entity_embedding = self.entity_embedding_layer(True)
        self.relation_embedding = self.relation_embedding_layer(True)

        # normalize entity embeddings...
        self.entity_embedding = tf.cond(tf.equal(self.train_mode, True),
                        lambda: tf.nn.l2_normalize(self.entity_embedding, axis=1), lambda: self.entity_embedding)

        # get embeddings ...
        self.s_ent_emb = tf.nn.embedding_lookup(self.entity_embedding, self.s_ent)
        self.r_rel_emb = tf.nn.embedding_lookup(self.relation_embedding, self.r_rel)
        self.t_ent_emb = tf.nn.embedding_lookup(self.entity_embedding, self.t_ent)

        triple_emb = self.s_ent_emb * self.r_rel_emb * self.t_ent_emb   #  None x 1 x 3
        self.pos_score = tf.reduce_sum(triple_emb, axis=2)  # None x 1
                                                            #  tf.nn.l2_normalize( tf.nn.l2_normalize()  for cosine_dist

    def prediction_layer(self):
        # prediction layer ...
        self.cand_entset_size = tf.compat.v1.placeholder(tf.int32, name='cand_ent_size')

        # select candidate_embedding ...
        cand_ent_ids = tf.reshape(tf.range(self.cand_entset_size), (1, self.cand_entset_size))  # 1 x num_cand_ent
        cand_ent_emb = tf.nn.embedding_lookup(self.entity_embedding, cand_ent_ids)       # 1 x num_cand_ent x 3
        # reshaping and tiling ..for batch process..... q_vec is None x 1  x 3....
        # so. cand_ent_emb should be ---> None x num_cand_ent x 3
        # thus, replicate 1 x num_cand_ent x 3 for None times in axis = 0..
        cand_ent_emb_tiled = tf.tile(cand_ent_emb, [tf.shape(self.r_rel_emb)[0], 1, 1])  # tiling along dim = 0 ----> None x #c x 3

        # reshaping and tiling neg_s prediction ..
        pred_score_s = tf.multiply(cand_ent_emb_tiled, self.r_rel_emb)   # None x #c x 3 * None x 1 x 3  --> None x #c x 3
        # None x #c x 3 *  None x 3 x 3 ---> None x #c x 3
        pred_score_s = tf.matmul(pred_score_s, self.t_ent_emb, transpose_b=True)  # None x #c  x 3 *  None x 1 x 3 ---> None x #c x 1
        self.pred_scores_s = tf.reshape(pred_score_s, [tf.shape(pred_score_s)[0], tf.shape(pred_score_s)[1]])  # None x #c

        # reshaping and tiling neg_t prediction ..
        t_q_emb_vec = self.s_ent_emb * self.r_rel_emb  # None x 1 x 3
        t_pred_score = tf.matmul(t_q_emb_vec, cand_ent_emb_tiled, transpose_b=True)
                                                              # None x 1  x 3 *  None x #c  x 3 ---> None x 1 x #c
        self.pred_scores_t = tf.reshape(t_pred_score, [tf.shape(t_pred_score)[0], tf.shape(t_pred_score)[2]])  # None x #c

    def train(self, input_batch_s, input_batch_r, input_batch_t, input_batch_s_neg, input_batch_t_neg, lr):
        return self.sess.run([self.optimize, self.loss], feed_dict={
            self.s_ent: input_batch_s,
            self.r_rel: input_batch_r,
            self.t_ent: input_batch_t,
            self.s_ent_neg: input_batch_s_neg,
            self.t_ent_neg: input_batch_t_neg,
            self.learning_rate: lr,
            self.train_mode: True
        })

    def evaluate(self, input_batch_s, input_batch_r, input_batch_t, input_batch_s_neg, input_batch_t_neg, test_ent_size):
        return self.sess.run([self.loss, self.pred_scores_s, self.pred_scores_t], feed_dict={
            self.s_ent: input_batch_s,
            self.r_rel: input_batch_r,
            self.t_ent: input_batch_t,
            self.s_ent_neg: input_batch_s_neg,
            self.t_ent_neg: input_batch_t_neg,
            self.cand_entset_size: test_ent_size,
            self.train_mode: False
        })

    def predict_target(self, input_s, input_r, test_ent_size):
        return self.sess.run(self.pred_scores_t, feed_dict={
            self.s_ent: input_s,
            self.r_rel: input_r,
            self.cand_entset_size: test_ent_size,
            self.train_mode: False
        })

    def predict_source(self, input_r, input_t, test_ent_size):
        return self.sess.run(self.pred_scores_s, feed_dict={
            self.r_rel: input_r,
            self.t_ent: input_t,
            self.cand_entset_size: test_ent_size,
            self.train_mode: False
        })

    def initialize_embedding(self, embedding):
        self.sess.run(self.embedding_init, feed_dict={
            self.embedding_placeholder: embedding
        })
