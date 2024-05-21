import os
import torch
import embeddings
from cupy_utils import *
from torch import nn
import argparse
import inspect
from torch import optim
from wd import wd
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cupy as cp
import re
import sys
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.memory.malloc_managed).malloc)

try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False

def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params

def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    # if FAISS_AVAILABLE:
    #     emb = emb.cpu().detach().numpy()
    #     query = query.cpu().detach().numpy()
    #     if hasattr(faiss, 'StandardGpuResources'):
    #         # gpu mode
    #         res = faiss.StandardGpuResources()
    #         config = faiss.GpuIndexFlatConfig()
    #         config.device = 0
    #         index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
    #     else:
    #         # cpu mode
    #         index = faiss.IndexFlatIP(emb.shape[1])
    #     index.add(emb)
    #     distances, _ = index.search(query, knn)
    #     return distances.mean(1)
    # else:
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
        torch.cuda.empty_cache()
    all_distances = torch.cat(all_distances)
    return all_distances#.numpy()

def get_candidates(emb1, emb2):
    """
    Get best translation pairs candidates.
    """
    bs = 128
    all_targets = []
    all_scores = []

    # number of source words to consider
    n_src = emb1.size(0)
    knn = 10

    average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)

    # for every source word
    for i in range(0, n_src, bs):

        # compute target words scores
        scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)
        del scores
        torch.cuda.empty_cache()
        # update scores / potential targets
        all_targets.append(best_targets.cpu())
        all_scores.append(best_scores.cpu())

    all_targets = torch.cat(all_targets, 0)
    all_scores = torch.cat(all_scores, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:,0].unsqueeze(1)
    ], 1)


    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_pairs = all_pairs[reordered]

    # sanity check
    assert all_pairs.size() == (n_src, 2)

    return all_pairs

def csls(emb1, emb2):
    average_dist1 = get_nn_avg_dist(emb2, emb1, 10)
    average_dist2 = get_nn_avg_dist(emb1, emb2, 10)
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)

    n_src = emb1.size(0)
    bs = 128
    all_scores = torch.empty(0, n_src).cuda()
    # for every source word
    for i in range(0, n_src, bs):
        # compute target words scores
        scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1).cuda()
        scores.mul_(2)
        scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
        all_scores = torch.cat((all_scores, scores), dim=0)
        del scores
        torch.cuda.empty_cache()
    return all_scores

def select_node(emb1, thershold,num):
    all_scores = csls(emb1, emb1)
    weight = {}

    bs = 128
    above_threshold_indices = torch.empty(0, 2).long().cuda()
    for i in range(0, emb1.size(0), bs):
        scores = all_scores[i:min(emb1.size(0), i + bs)]
        indices = torch.nonzero(scores > thershold)
        indices[:,0] = indices[:,0] + i
        above_threshold_indices = torch.cat((above_threshold_indices, indices), dim=0)
        del indices, scores
        torch.cuda.empty_cache()


    for row, col in above_threshold_indices:
        if int(row) in weight.keys():
            weight[int(row)] = weight[int(row)] + float(all_scores[row][col])
        else:
            weight.update({int(row): float(all_scores[row][col])})

    del all_scores
    torch.cuda.empty_cache()

    sorted_weight_desc = dict(sorted(weight.items(), key=lambda item: item[1], reverse=True))

    if num ==0 :
        num = len(sorted_weight_desc)

    print(len(sorted_weight_desc))


    return list(sorted_weight_desc.keys())[:num]

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    nearest_words = []
    for i, idx in enumerate(k_best):
        nearest_words.append(tgt_id2word[idx])
    return scores, nearest_words

def evaluate(args, emb1, emb2, src_word2ind, src_ind2word, trg_ind2word):
    # file_lexicon = open(f'/home/xym/..Multi-adversarial/wordsim/MUSEdictionaries/{args.src_lang}-{args.trg_lang}.5000-6500.txt', 'r', encoding='utf-8')
    file_lexicon = open(
        f'/data/hl/BLI/acl2024-bli/hl/clwe/dictionaries/{args.src_lang}-{args.trg_lang}.5000-6500.txt', 'r',
        encoding='utf-8')
    src_gold_words = []
    trg_gold_words = []
    binary_lexicon = {}
    number_of_nearest_words = 5  # 映射的前几个单词，需要修改

    for line in file_lexicon.readlines():
        line = line.rstrip("\n")
        line = line.replace("\t", " ")
        src_word, trg_word = line.split(' ')
        if src_word not in binary_lexicon:
            binary_lexicon[src_word] = [trg_word]
        else:
            binary_lexicon[src_word].append(trg_word)
        # print(src_word,",",trg_word)
        src_gold_words.append(src_word)
        trg_gold_words.append(trg_word)

    # print(binary_lexicon)
    file_lexicon.close()

    # find the nearest neighbors for each word
    count = 0
    hit_count = 0

    for key, value in binary_lexicon.items():
        src_gold_word = key
        trg_gold_words = value
        # if the word in the embedding space
        if src_gold_word in src_word2ind.keys():
            hit_count = hit_count + 1
            # trg_gold_words = binary_lexicon.get(src_gold_word)
            similar_scores, nearest_words = get_nn(src_gold_word, emb1, src_ind2word, emb2,
                                                   trg_ind2word, number_of_nearest_words)

            for candidate_word in nearest_words:
                if candidate_word in trg_gold_words:
                    count = count + 1
                    print(count, "hit:", src_gold_word, candidate_word)
                    break

    acc1 = count / len(binary_lexicon)
    print('Acc: {0:.4f}'.format(acc1))
    print(f"在双语词典中有{hit_count}个单词也在构建的词典中")
    print(f"双语词典和构建词典的{hit_count}个单词中，在最近邻为{number_of_nearest_words}找到了{count}个")
    acc2 = count / hit_count
    print('Acc: {0:.4f}'.format(acc2))

def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m*mask

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def left_fill(args, all_pairs, new_src_emb_words_alreay, src_word2ind, src_ind2word, trg_ind2word, n_clusters, subspace_dico_new,x_tensor,mapping_wx, srcfile):
    if len(new_src_emb_words_alreay) != 75000:
        id_s = [src_word2ind[s_word] for s_word in new_src_emb_words_alreay]
        not_in_list = list(set([k for k in range(75000)]) - set(id_s))
        print(len(not_in_list))

        s_sub_left = {}
        for pair in all_pairs:
            if int(pair[0]) in not_in_list:
                sword = src_ind2word[float(pair[0])]
                tword = trg_ind2word[float(pair[1])]
                num = subspace_dico_new[tword]
                if num not in s_sub_left:
                    s_sub_left.update({num: [sword]})
                else:
                    s_sub_left[num].append(sword)

        for j in range(n_clusters):
            if j in s_sub_left.keys():
                left_id_s = [src_word2ind[s_word] for s_word in s_sub_left[j]]
                print(f"#####{j}:{len(left_id_s)}")
                left_src_emb_select = torch.index_select(x_tensor, 0, torch.tensor(left_id_s).cuda())
                to_reload_wx = cp.asnumpy(torch.load(os.path.join(args.sub_mappingpath, f'best_mapping_wx_{j}.pth')))
                # to_reload_wx = cp.asnumpy(torch.load(os.path.join(args.mappingpath, f'{args.src_lang}-{args.trg_lang}-best_mapping_wx.pth')))
                to_reload_wx = torch.from_numpy(to_reload_wx).cuda()
                mapping_wx.weight.data.copy_(to_reload_wx.type_as(mapping_wx.weight.data))
                left_src_emb_mapped = mapping_wx(left_src_emb_select)
                embeddings.write(s_sub_left[j], cp.asarray(left_src_emb_mapped.detach().cpu().numpy()), srcfile)
                del left_src_emb_select, left_src_emb_mapped

def cl_loss(embed_src, embed_pos, scale=20.0):
    '''
    scale is a temperature parameter
    '''
    embed_neg = None
    if embed_neg is not None:
        embed_pos = torch.cat([embed_pos, embed_neg], dim=0)

    scores = cos_sim(embed_src, embed_pos) * scale

    labels = torch.tensor(range(len(scores)),
                          dtype=torch.long,
                          device=scores.device)  # Example a[i] should match with b[i]

    loss = nn.CrossEntropyLoss()

    return loss(scores, labels)

def cos_sim(a, b):
    """ the function is same with torch.nn.F.cosine_similarity but processed the problem of tensor dimension
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def main():
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('--src_input', help='the input source embeddings')
    parser.add_argument('--trg_input', help='the input target embeddings')
    parser.add_argument('--src_lang', default='ru', help='source language')
    parser.add_argument('--trg_lang', default='en', help='target language')
    parser.add_argument('--src_output', help='the output source embeddings')
    parser.add_argument('--trg_output', help='the output target embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')
    parser.add_argument('--mappingpath', type=str, default='mapping', help='path to save the optimal mapping path')
    parser.add_argument('--sub_mappingpath', type=str, default='sub_best_mapping',
                        help='path to save the optimal mapping path')
    parser.add_argument('--hyper', type=float, default=0.60)
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--supervised', metavar='DICTIONARY', help='recommended if you have a large training dictionary')
    recommended_type.add_argument('--semi_supervised', metavar='DICTIONARY', help='recommended if you have a small seed dictionary')
    recommended_type.add_argument('--identical', action='store_true', help='recommended if you have no seed dictionary but can rely on identical words')
    recommended_type.add_argument('--unsupervised', action='store_true', help='recommended if you have no seed dictionary and do not want to rely on identical words')
    recommended_type.add_argument('--acl2018', action='store_true', help='reproduce our ACL 2018 system')
    recommended_type.add_argument('--aaai2018', metavar='DICTIONARY', help='reproduce our AAAI 2018 system')
    recommended_type.add_argument('--acl2017', action='store_true', help='reproduce our ACL 2017 system with numeral initialization')
    recommended_type.add_argument('--acl2017_seed', metavar='DICTIONARY', help='reproduce our ACL 2017 system with a seed dictionary')
    recommended_type.add_argument('--emnlp2016', metavar='DICTIONARY', help='reproduce our EMNLP 2016 system')

    init_group = parser.add_argument_group('advanced initialization arguments', 'Advanced initialization arguments')
    init_type = init_group.add_mutually_exclusive_group()
    init_type.add_argument('-d', '--init_dictionary', default=sys.stdin.fileno(), metavar='DICTIONARY', help='the training dictionary file (defaults to stdin)')
    init_type.add_argument('--init_identical', action='store_true', help='use identical words as the seed dictionary')
    init_type.add_argument('--init_numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    init_type.add_argument('--init_unsupervised', action='store_true', help='use unsupervised initialization')
    init_group.add_argument('--unsupervised_vocab', type=int, default=0, help='restrict the vocabulary to the top k entries for unsupervised initialization')

    mapping_group = parser.add_argument_group('advanced mapping arguments', 'Advanced embedding mapping arguments')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=[], help='the normalization actions to perform in order')
    mapping_group.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    mapping_group.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the source language embeddings')
    mapping_group.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the target language embeddings')
    mapping_group.add_argument('--src_dewhiten', choices=['src', 'trg'], help='de-whiten the source language embeddings')
    mapping_group.add_argument('--trg_dewhiten', choices=['src', 'trg'], help='de-whiten the target language embeddings')
    mapping_group.add_argument('--dim_reduction', type=int, default=0, help='apply dimensionality reduction')
    mapping_type = mapping_group.add_mutually_exclusive_group()
    mapping_type.add_argument('-c', '--orthogonal', action='store_true', help='use orthogonal constrained mapping')
    mapping_type.add_argument('-u', '--unconstrained', action='store_true', help='use unconstrained mapping')

    self_learning_group = parser.add_argument_group('advanced self-learning arguments', 'Advanced arguments for self-learning')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--vocabulary_cutoff', type=int, default=0, help='restrict the vocabulary to the top k entries')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='the direction for dictionary induction (defaults to union)')
    self_learning_group.add_argument('--csls', type=int, nargs='?', default=0, const=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, metavar='DICTIONARY', help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--stochastic_initial', default=0.1, type=float, help='initial keep probability stochastic dictionary induction (defaults to 0.1)')
    self_learning_group.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    self_learning_group.add_argument('--stochastic_interval', default=50, type=int, help='stochastic dictionary induction interval (defaults to 50)')
    self_learning_group.add_argument('--log', default=r'./log/en-ja.txt',help='write to a log file in tsv format at each iteration')
    self_learning_group.add_argument('-v', '--verbose', action='store_true', default = True, help='write log information to stderr at each iteration')
    args = parser.parse_args()

    if args.supervised is not None:
        parser.set_defaults(init_dictionary=args.supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.semi_supervised is not None:
        parser.set_defaults(init_dictionary=args.semi_supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.identical:
        parser.set_defaults(init_identical=True, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.unsupervised or args.acl2018:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.aaai2018:
        parser.set_defaults(init_dictionary=args.aaai2018, normalize=['unit', 'center'], whiten=True, trg_reweight=1, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.acl2017:
        parser.set_defaults(init_numerals=True, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.acl2017_seed:
        parser.set_defaults(init_dictionary=args.acl2017_seed, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.emnlp2016:
        parser.set_defaults(init_dictionary=args.emnlp2016, orthogonal=True, normalize=['unit', 'center'], batch_size=1000)
    args = parser.parse_args()

    path_wx_mapfile = os.path.join(args.mappingpath, f'{args.src_lang}-{args.trg_lang}-best_mapping_wx.pth')
    path_wz_mapfile = os.path.join(args.mappingpath, f'{args.src_lang}-{args.trg_lang}-best_mapping_wz.pth')


    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    b_xent = nn.BCEWithLogitsLoss()
    # Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    print(len(src_words))
    print(x.shape)

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:

        xp = np
    xp.random.seed(args.seed)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    src_ind2word = {i: word for i, word in enumerate(src_words)}
    trg_ind2word = {i: word for i, word in enumerate(trg_words)}

    # STEP 0: Normalization
    embeddings.normalize(x, args.normalize)
    embeddings.normalize(z, args.normalize)

    # cluster for english words
    n_clusters = 9
    # 第一次跑，需要跑一遍聚类，需要用到以下代码
    matrix = cp.asnumpy(z)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    data = {
        'words': trg_words,
        'Cluster': kmeans.labels_
    }
    df = pd.DataFrame(data)
    df.to_csv(f'cluster_output_kmeans-{n_clusters}-75000-new.csv', index=False)
    src_indices = []
    trg_indices = []
    if args.init_unsupervised:
        sim_size = min(x.shape[0], z.shape[0]) if args.unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0],
                                                                                        args.unsupervised_vocab)
        u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u * s).dot(u.T)
        u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u * s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, args.normalize)
        embeddings.normalize(zsim, args.normalize)
        sim = xsim.dot(zsim.T)
        if args.csls_neighborhood > 0:
            knn_sim_fwd = topk_mean(sim, k=args.csls_neighborhood)
            knn_sim_bwd = topk_mean(sim.T, k=args.csls_neighborhood)
            sim -= knn_sim_fwd[:, xp.newaxis] / 2 + knn_sim_bwd / 2
        if args.direction == 'forward':
            src_indices = xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif args.direction == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = xp.arange(sim_size)
        elif args.direction == 'union':
            src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
            trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
        del xsim, zsim, sim
    else:
        f = open(args.init_dictionary, encoding=args.encoding, errors='surrogateescape')
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)


    # Create log file
    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')

    # Allocate memory
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)
    src_size = x.shape[0] if args.vocabulary_cutoff <= 0 else min(x.shape[0], args.vocabulary_cutoff)
    trg_size = z.shape[0] if args.vocabulary_cutoff <= 0 else min(z.shape[0], args.vocabulary_cutoff)
    simfwd = xp.empty((args.batch_size, trg_size), dtype=dtype)
    simbwd = xp.empty((args.batch_size, src_size), dtype=dtype)

    best_sim_forward = xp.full(src_size, -100, dtype=dtype)
    src_indices_forward = xp.arange(src_size)
    trg_indices_forward = xp.zeros(src_size, dtype=int)
    best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
    src_indices_backward = xp.zeros(trg_size, dtype=int)
    trg_indices_backward = xp.arange(trg_size)
    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)

    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = args.stochastic_initial
    t = time.time()
    end = not args.self_learning
    while True:

        # Increase the keep probability if we have not improve in args.stochastic_interval iterations
        if it - last_improvement > args.stochastic_interval:
            if keep_prob >= 1.0:
                end = True
            keep_prob = min(1.0, args.stochastic_multiplier * keep_prob)
            last_improvement = it

        # Update the embedding mapping
        if args.orthogonal or not end:  # orthogonal mapping
            u, s, vt = xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
            w = vt.T.dot(u.T)
            x.dot(w, out=xw)
            zw[:] = z
        elif args.unconstrained:  # unconstrained mapping
            x_pseudoinv = xp.linalg.inv(x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
            w = x_pseudoinv.dot(z[trg_indices])
            x.dot(w, out=xw)
            zw[:] = z
        else:  # advanced mapping
            print("**********************")
            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z

            # STEP 1: Whitening
            def whitening_transformation(m):
                u, s, vt = xp.linalg.svd(m, full_matrices=False)
                return vt.T.dot(xp.diag(1 / s)).dot(vt)

            if args.whiten:
                wx1 = whitening_transformation(xw[src_indices])
                wz1 = whitening_transformation(zw[trg_indices])
                xw = xw.dot(wx1)
                zw = zw.dot(wz1)

            # STEP 2: Orthogonal mapping
            wx2, s, wz2_t = xp.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
            wz2 = wz2_t.T
            xw = xw.dot(wx2)
            zw = zw.dot(wz2)

            # STEP 3: Re-weighting
            xw *= s ** args.src_reweight
            zw *= s ** args.trg_reweight

            # STEP 4: De-whitening
            if args.src_dewhiten == 'src':
                xw = xw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.src_dewhiten == 'trg':
                xw = xw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))
            if args.trg_dewhiten == 'src':
                zw = zw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.trg_dewhiten == 'trg':
                zw = zw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))

            # STEP 5: Dimensionality reduction
            if args.dim_reduction > 0:
                xw = xw[:, :args.dim_reduction]
                zw = zw[:, :args.dim_reduction]

        # Self-learning
        if end:
            break
        else:
            # Update the training dictionary
            if args.direction in ('forward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j - i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
                    simfwd[:j - i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j - i] -= knn_sim_bwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simfwd[:j - i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
            if args.direction in ('backward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j - i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
                    simbwd[:j - i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j - i] -= knn_sim_fwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simbwd[:j - i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
            if args.direction == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif args.direction == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif args.direction == 'union':
                src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            if args.direction == 'forward':
                objective = xp.mean(best_sim_forward).tolist()
            elif args.direction == 'backward':
                objective = xp.mean(best_sim_backward).tolist()
            elif args.direction == 'union':
                objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= args.threshold:
                last_improvement = it
                best_objective = objective

            # Logging
            duration = time.time() - t
            if args.verbose:
                print(file=sys.stderr)
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
                print('\t- Drop probability: {0:9.4f}%'.format(100 - 100 * keep_prob), file=sys.stderr)
                sys.stderr.flush()
                print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, '', duration), file=log)
                log.flush()

        t = time.time()
        it += 1
    # save the best mapping
    w_x = ((wx1.dot(wx2)) * (s ** args.src_reweight)).dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
    w_z = ((wz1.dot(wz2)) * (s ** args.trg_reweight)).dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))
    w_x = torch.from_numpy(cp.asnumpy(w_x)).float()
    w_z = torch.from_numpy(cp.asnumpy(w_z)).float()
    torch.save(w_x, path_wx_mapfile)
    torch.save(w_z, path_wz_mapfile)
    df = pd.read_csv(f'cluster_output_kmeans-{n_clusters}-75000-new.csv', keep_default_na=False)

    # get target(en) subspaces
    t_sub = {}
    for index, row in df.iterrows():
        num = row['Cluster']
        if num not in t_sub:
            t_sub.update({num: [row['words']]})
        else:
            t_sub[num].append(row['words'])

    subspace_dico_new = {}
    for index, row in df.iterrows():
        key = row["words"]
        value = row["Cluster"]
        subspace_dico_new.update({f"{key}": value})


    mapping_wx = nn.Linear(300, 300, bias=False)
    mapping_wx = mapping_wx.to(torch.device(x.device.id))
    mapping_wz = nn.Linear(300, 300, bias=False)
    mapping_wz = mapping_wz.to(torch.device(z.device.id))

    x_tensor = torch.from_numpy(cp.asnumpy(x)).cuda()
    z_tensor = torch.from_numpy(cp.asnumpy(z)).cuda()
    del x, z
    torch.cuda.empty_cache()


    # initialization
    to_reload_wx = torch.from_numpy(cp.asnumpy(torch.load(path_wx_mapfile))).cuda()
    to_reload_wz = torch.from_numpy(cp.asnumpy(torch.load(path_wz_mapfile))).cuda()
    mapping_wx.weight.data.copy_(to_reload_wx.t().type_as(mapping_wx.weight.data))
    mapping_wz.weight.data.copy_(to_reload_wz.t().type_as(mapping_wz.weight.data))

    x_mapped = mapping_wx(x_tensor)
    z_mapped = mapping_wz(z_tensor)

    srcfile = open(args.src_output, mode='a', encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_output, mode='a', encoding=args.encoding, errors='surrogateescape')



    # optimizer
    optim_fn, optim_params = get_optimizer(args.map_optimizer)
    mapping_wx_optimizer = optim_fn(mapping_wx.parameters(), **optim_params)
    mapping_wz_optimizer = optim_fn(mapping_wz.parameters(), **optim_params)


    all_pairs = get_candidates(x_mapped, z_mapped)

    s_sub = {}
    for pair in all_pairs:
        sword = src_ind2word[float(pair[0])]
        tword = trg_ind2word[float(pair[1])]
        num = subspace_dico_new[tword]
        if num not in s_sub:
            s_sub.update({num: [sword]})
        else:
            s_sub[num].append(sword)

    # 根据字典值的长度进行降序排序
    s_sub_by_length_desc = dict(sorted(s_sub.items(), key=lambda item: len(item[1]), reverse=True))

    already = []
    new_src_emb_words_alreay = []

    for i in s_sub_by_length_desc.keys():
        mapping_wx.weight.data.copy_(to_reload_wx.t().type_as(mapping_wx.weight.data))
        mapping_wz.weight.data.copy_(to_reload_wz.t().type_as(mapping_wz.weight.data))
        path_wx_mapfile_update = os.path.join(args.sub_mappingpath, f'best_mapping_wx_{i}.pth')
        path_wz_mapfile_update = os.path.join(args.sub_mappingpath, f'best_mapping_wz_{i}.pth')
        id_t = [trg_word2ind[t_word] for t_word in t_sub[i]]
        print(f"##TRG-{i}: {len(id_t)}")
        trg_emb_select = torch.index_select(z_tensor, 0, torch.tensor(id_t).cuda())

        stablity = 0
        iter = 0
        #### the original version is 0.95
        while stablity < 0.85:
            loss_best = 100
            count = 0
            loss_iter = 0
            while count <3 :
                id_s = [src_word2ind[s_word] for s_word in s_sub[i]]
                id_alr = [src_word2ind[s_word] for s_word in new_src_emb_words_alreay]
                id_s = list(set(id_s)- (set(id_s)&set(id_alr)))
                print(f"##SRC-{i}: {len(id_s)}")
                src_emb_select = torch.index_select(x_tensor, 0, torch.tensor(id_s).cuda())
                if iter == 0 and loss_iter == 0:
                    src_emb_mapped = torch.index_select(x_mapped, 0, torch.tensor(id_s).cuda())
                    trg_emb_mapped = torch.index_select(z_mapped, 0, torch.tensor(id_t).cuda())
                    print("*"*30)
                else:
                    src_emb_mapped = torch.index_select(new_x_mapped, 0, torch.tensor(id_s).cuda())
                    trg_emb_mapped = torch.index_select(new_z_mapped, 0, torch.tensor(id_t).cuda())

                # w-distance
                nodes_s = select_node(src_emb_mapped, args.hyper, 0)
                nodes_t = select_node(trg_emb_mapped, args.hyper, 0)

                important_src_emb_mapped = torch.index_select(src_emb_mapped, 0, torch.tensor(nodes_s).cuda())
                important_trg_emb_mapped = torch.index_select(trg_emb_mapped, 0, torch.tensor(nodes_t).cuda())

                select_pairs = get_candidates(src_emb_mapped, trg_emb_mapped)
                v_limit = int(0.20 * len(select_pairs))
                select_pairs = select_pairs[:v_limit]
                # print(select_pairs)
                cl_important_src_emb_mapped = torch.index_select(src_emb_mapped, 0, torch.tensor(select_pairs[:,0]).cuda())
                cl_important_trg_emb_mapped_pos = torch.index_select(trg_emb_mapped, 0, torch.tensor(select_pairs[:,1]).cuda())

                contrast_loss = cl_loss(cl_important_src_emb_mapped, cl_important_trg_emb_mapped_pos)

                uncommon_elements = list(set([k for k in range(n_clusters)]) - set([i]))
                negetive_distance_s2t = torch.empty((0, 1)).cuda()
                for elem in uncommon_elements:
                    select_id = [trg_word2ind[t_word] for t_word in t_sub[elem]]
                    if elem in already:
                        select = torch.index_select(new_z_mapped, 0, torch.tensor(select_id).cuda())
                        ###################!!!!修改过
                        nodes_select = select_node(select, args.hyper,v_limit)
                        # nodes_select = select_node(select, args.hyper, 0)
                        select_mapped = torch.index_select(select, 0, torch.tensor(nodes_select).cuda())
                    else:
                        select = torch.index_select(z_mapped, 0, torch.tensor(select_id).cuda())
                        ###################!!!!修改过
                        nodes_select = select_node(select, args.hyper,v_limit)
                        # nodes_select = select_node(select, args.hyper, 0)
                        select_mapped =torch.index_select(select, 0, torch.tensor(nodes_select).cuda())
                    wdistance_neg_s2t = wd(important_src_emb_mapped.unsqueeze(0), select_mapped.unsqueeze(0), 0.5)
                    negetive_distance_s2t = torch.cat((negetive_distance_s2t, wdistance_neg_s2t), dim=0)
                wdistance_pos_s2t = wd(important_src_emb_mapped.unsqueeze(0), important_trg_emb_mapped.unsqueeze(0), 0.5)
                wdistance_s2t = torch.cat((wdistance_pos_s2t, negetive_distance_s2t), dim=0)
                logits_s2t = torch.exp(-wdistance_s2t)
                lbl_1 = torch.ones(1).cuda()
                lbl_2 = torch.zeros(n_clusters-1).cuda()
                lbl = torch.cat((lbl_1, lbl_2), 0).cuda()
                loss_wd_s2t = b_xent(torch.squeeze(logits_s2t), lbl)

                negetive_distance_t2s = torch.empty((0, 1)).cuda()
                for elem in uncommon_elements:
                    select_id = [src_word2ind[s_word] for s_word in s_sub[elem]]
                    if elem in already:
                        select = torch.index_select(new_x_mapped, 0, torch.tensor(select_id).cuda())
                        nodes_select = select_node(select,  args.hyper, v_limit)
                        select_mapped = torch.index_select(select, 0, torch.tensor(nodes_select).cuda())
                    else:
                        select = torch.index_select(x_mapped, 0, torch.tensor(select_id).cuda())
                        nodes_select = select_node(select,  args.hyper, v_limit)
                        select_mapped = torch.index_select(select, 0, torch.tensor(nodes_select).cuda())
                    wdistance_neg_t2s = wd(important_trg_emb_mapped.unsqueeze(0), select_mapped.unsqueeze(0), 0.5)
                    negetive_distance_t2s = torch.cat((negetive_distance_t2s, wdistance_neg_t2s), dim=0)

                wdistance_pos_t2s = wd(important_trg_emb_mapped.unsqueeze(0), important_src_emb_mapped.unsqueeze(0),
                                       0.5)
                wdistance_t2s = torch.cat((wdistance_pos_t2s, negetive_distance_t2s), dim=0)
                logits_t2s = torch.exp(-wdistance_t2s)
                loss_wd_t2s = b_xent(torch.squeeze(logits_t2s), lbl)



                print(loss_wd_s2t, loss_wd_t2s, contrast_loss)
                lamda = float(((loss_wd_s2t + loss_wd_t2s)/2))/float(contrast_loss)
                loss = (loss_wd_s2t + loss_wd_t2s)/2 + contrast_loss

                print("####loss:", loss)

                mapping_wx_optimizer.zero_grad()
                mapping_wz_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                mapping_wx_optimizer.step()
                mapping_wz_optimizer.step()
                loss_iter += 1

                del  select, nodes_select, select_mapped
                torch.cuda.empty_cache()

                if loss < loss_best:
                    loss_best = loss
                    wx_best = mapping_wx.weight.data
                    wz_best = mapping_wz.weight.data
                    if len(already) == 0:
                        new_x_mapped = x_mapped
                        new_z_mapped = z_mapped
                        new_x_select_mapped = mapping_wx(src_emb_select)
                        new_x_mapped[id_s, :] = new_x_select_mapped
                        new_z_select_mapped = mapping_wz(trg_emb_select)
                        new_z_mapped[id_t, :] = new_z_select_mapped
                        del new_x_select_mapped, new_z_select_mapped
                    else:
                        new_x_select_mapped = mapping_wx(src_emb_select)
                        new_x_mapped[id_s, :] = new_x_select_mapped
                        new_z_select_mapped = mapping_wz(trg_emb_select)
                        new_z_mapped[id_t, :] = new_z_select_mapped
                        del new_x_select_mapped, new_z_select_mapped
                    count = 0
                else:
                    count += 1

                del src_emb_select
                torch.cuda.empty_cache()

            mapping_wx.weight.data.copy_(wx_best.type_as(mapping_wx.weight.data))
            mapping_wz.weight.data.copy_(wz_best.type_as(mapping_wz.weight.data))

            del src_emb_mapped, trg_emb_mapped, important_src_emb_mapped, important_trg_emb_mapped
            torch.cuda.empty_cache()

            new_pairs = get_candidates(new_x_mapped, new_z_mapped)

            new_src_words = []
            for pair in new_pairs:
                sword = src_ind2word[float(pair[0])]
                if sword not in new_src_emb_words_alreay:
                    tword = trg_ind2word[float(pair[1])]
                    num = subspace_dico_new[tword]
                    # twords = [trg_ind2word[float(_id)] for _id in pair[1:]]
                    # num_list = [subspace_dico_new[tword] for tword in twords]
                    # count = Counter(num_list)
                    # num, max_count = count.most_common(1)[0]
                    if num == i:
                        new_src_words.append(sword)


            common_elements = set(s_sub[i]) & set(new_src_words)
            stablity = len(common_elements) / len(set(s_sub[i]))
            print(stablity)
            s_sub[i] = new_src_words
            iter += 1

        already.append(i)
        new_src_emb_words_alreay += s_sub[i]
        print(f"###{i}: {len(s_sub[i])}")
        print(len(new_src_emb_words_alreay))

        evaluate(args, new_x_mapped.detach().cpu().numpy(), new_z_mapped.detach().cpu().numpy(), src_word2ind, src_ind2word, trg_ind2word)

        new_id_s = [src_word2ind[s_word] for s_word in s_sub[i]]
        new_src_emb_select = torch.index_select(x_tensor, 0, torch.tensor(new_id_s).cuda())
        new_src_emb_mapped = mapping_wx(new_src_emb_select)
        new_trg_emb_mapped = mapping_wz(trg_emb_select)
        new_x_mapped[new_id_s, :] = new_src_emb_mapped
        new_z_mapped[id_t, :] = new_trg_emb_mapped
        embeddings.write(s_sub[i], cp.asarray(new_src_emb_mapped.detach().cpu().numpy()), srcfile)
        embeddings.write(t_sub[i], cp.asarray(new_trg_emb_mapped.detach().cpu().numpy()), trgfile)
        w_x_update = mapping_wx.weight.data.float()
        torch.save(w_x_update, path_wx_mapfile_update)
        w_z_update = mapping_wz.weight.data.float()
        torch.save(w_z_update, path_wz_mapfile_update)
        del new_src_emb_select, new_src_emb_mapped, new_id_s, w_x_update, w_z_update, trg_emb_select
        torch.cuda.empty_cache()

    print(len(new_src_emb_words_alreay))
    left_fill(args, all_pairs, new_src_emb_words_alreay, src_word2ind, src_ind2word, trg_ind2word, n_clusters,subspace_dico_new,x_tensor,mapping_wx, srcfile)



if __name__ == '__main__':
    main()