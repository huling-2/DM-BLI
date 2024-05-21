import io
import numpy as np
import argparse
import os
import torch
import sys

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



def load_vec(emb_path, nmax): #50000
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            a = line.rstrip().split(' ')
            if len(a) != 2:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                assert word not in word2id, 'word found twice'
                vectors.append(vect)
                word2id[word] = len(word2id)
            if len(word2id) == nmax:
                    break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


parser = argparse.ArgumentParser(description='loading')
parser.add_argument("--src_lang", type=str, default='zh', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='en', help="Target language")
parser.add_argument("--src_path", type=str, default='ZH2EN-vectors_vecmap_zh-en-0.8-0.8.txt', help="path to mapped source embeddings")
parser.add_argument("--tgt_path", type=str, default='ZH2EN-vectors_vecmap_en-zh-0.8-0.8.txt', help="path to mapped target embeddings")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=75000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--nmax", type=int, default=75000, help="maximum number of word embeddings to load 50000")
parser.add_argument("--knn", type=int, default=5, help="the number of nearest neighbors to")


params = parser.parse_args()

print(f"--------{params.src_lang}---------")
src_embeddings, src_id2word, src_word2id = load_vec(params.src_path, params.nmax)
print(len(src_id2word))
print(f"--------{params.tgt_lang}---------")
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(params.tgt_path, params.nmax)
print(len(tgt_id2word))


# Get nearest neighbors

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    #print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    nearest_words = []
    for i, idx in enumerate(k_best):
        #print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        nearest_words.append(tgt_id2word[idx])
    return scores, nearest_words

def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    # if FAISS_AVAILABLE:
    #     emb = emb.cpu().numpy()
    #     query = query.cpu().numpy()
    #     if hasattr(faiss, 'StandardGpuResources'):
    #         # gpu mode
    #         res = faiss.StandardGpuResources()
    #         config = faiss.GpuIndexFlatConfig()
    #         config.device = 0
    #         index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
    #     else:
    #         # cpu mode
    #         index = faiss.IndexFlatIP(emb.shape[1])
    #     print(emb.type)
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

def get_candidates(emb1, emb2, number_of_nearest_words):
    """
    Get best translation pairs candidates.
    """
    bs = 128
    all_targets = []
    emb1 = torch.from_numpy(emb1)
    emb2 = torch.from_numpy(emb2)

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
        best_scores, best_targets = scores.topk(number_of_nearest_words, dim=1, largest=True, sorted=True)
        del scores
        torch.cuda.empty_cache()
        # update scores / potential targets
        all_targets.append(best_targets.cpu())


    all_targets = torch.cat(all_targets, 0)
    print(all_targets.size)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets], 1)

    # sanity check
    print(all_pairs.size())
    assert all_pairs.size() == (n_src, number_of_nearest_words+1)

    return all_pairs

#read binary dictionaries

file_lexicon = open(f'/data/hl/BLI/acl2024-bli/hl/clwe/dictionaries/{params.src_lang}-{params.tgt_lang}.txt', 'r', encoding='utf-8')
src_gold_words = []
trg_gold_words = []
binary_lexicon = {}
number_of_nearest_words = params.knn # 映射的前几个单词，需要修改

for line in file_lexicon.readlines():
    line = line.rstrip("\n")
    line = line.replace("\t"," ")
    src_word, trg_word = line.split(' ')
    #print(src_word)
    if src_word not in binary_lexicon:
        binary_lexicon[src_word] = [trg_word]
    else:
        binary_lexicon[src_word].append(trg_word)
    #print(src_word,",",trg_word)
    src_gold_words.append(src_word)
    trg_gold_words.append(trg_word)

# print(binary_lexicon)
file_lexicon.close()

#find the nearest neighbors for each word
count = 0
hit_count= 0
all_pairs = get_candidates(src_embeddings, tgt_embeddings, number_of_nearest_words)

for key, value in binary_lexicon.items():
    src_gold_word = key
    trg_gold_words = value
    # if the word in the embedding space
    if src_gold_word in src_word2id.keys():
        hit_count = hit_count+1
        # trg_gold_words = binary_lexicon.get(src_gold_word)
        # similar_scores, nearest_words = get_nn(src_gold_word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, number_of_nearest_words)
        src_wordid = src_word2id[src_gold_word]
        nearest_words = [tgt_id2word[int(i)] for i in all_pairs[src_wordid][1:]]
        for candidate_word in nearest_words:
            if candidate_word in trg_gold_words:
                count = count+1
                # print("hit:", src_gold_word, candidate_word)
                break
            #else:
                #print("Source:",src_gold_word, " predict:", candidate_word, " True:", trg_gold_words)


acc1 = count / len(binary_lexicon)
print('Acc: {0:.4f}'.format(acc1))
print(f"在双语词典中有{hit_count}个单词也在构建的词典中")
print(f"双语词典和构建词典的{hit_count}个单词中，在最近邻为{number_of_nearest_words}找到了{count}个")
acc2 = count / hit_count
print('Acc: {0:.4f}'.format(acc2))
