import os
import sys
lang_pairs = [
 # ('es', 'en', 0.6),
 # ('de', 'en', 0.6)
 # ('ru', 'en', 0.50),
 # ('bg', 'en', 0.50),
 # ('ar', 'en', 0.50),
 # ('tr', 'en', 0.50),
 # ('id', 'en', 0.51),
 # ('hi', 'en', 0.48),
 # ('ca', 'en', 0.55)
 #   ('zh', 'en', 0.47),
('ru', 'en', 0.50)
# ('bg', 'en', 0.50)


]

for (lang1, lang2, hyper) in lang_pairs:
    print(lang1, lang2, hyper)
    sys.stdout.flush()

    size_train = "5k" # or "1k"
    ROOT_EMB_SRC = "fasttext/wiki-{}-75000.vec".format(lang1)
    ROOT_EMB_TRG = "fasttext/wiki-{}-200000.vec".format(lang2)
    SRC_OUTPUT = 'mapped_embedding_sup/or_{}2{}-vectors_vecmap_{}-{}-new.txt'.format(lang1,lang2,lang1,lang2)
    TRG_OUTPUT = 'mapped_embedding_sup/or_{}2{}-vectors_vecmap_{}-{}-new.txt'.format(lang1,lang2,lang2,lang1)
    # ROOT_TEST_DICT = "/student15/hl/clwe/dictionaries/{}-{}.5000-6500.txt".format(lang1, lang2)
    ROOT_TRAIN_DICT = "dictionaries/{}-{}.0-5000.txt".format(lang1, lang2)
    # SAVE_ROOT = "/student15/hl/ContrastiveBLI-main/C1/SAVE_EMB" # save aligend WEs
    if lang1 == 'bg':
        SRC_OUTPUT = 'mapped_embedding/new_or_{}2{}-vectors_vecmap_{}-{}-new.txt'.format(lang1,lang2,lang1,lang2)
        TRG_OUTPUT = 'mapped_embedding/new_or_{}2{}-vectors_vecmap_{}-{}-new.txt'.format(lang1, lang2,lang2,lang1)
        os.system('CUDA_VISIBLE_DEVICES=0  python our_orth.py --unsupervised  --src_lang {} --trg_lang {} --src_input {} --trg_input {} --src_output {} --trg_output {} --hyper {} --cuda'.format(lang1, lang2, ROOT_EMB_SRC, ROOT_EMB_TRG, SRC_OUTPUT, TRG_OUTPUT, hyper))
    else:
        os.system('CUDA_VISIBLE_DEVICES=0  python our_orth.py --supervised {} --src_lang {} --trg_lang {} --src_input {} --trg_input {} --src_output {} --trg_output {} --hyper {} --cuda'.format(ROOT_TRAIN_DICT, lang1, lang2, ROOT_EMB_SRC, ROOT_EMB_TRG, SRC_OUTPUT, TRG_OUTPUT, hyper))