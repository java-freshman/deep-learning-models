import os

import numpy as np

import time
import pickle


prd_img_vec_path = 'extract_feature/embedding_vec'
prdid_name_path = 'extract_feature/img_name'
new_dcd = 'B43130701'
B43_newdcd_cate_prdid = pickle.load(open('B43_newdcd_cate_prdid', 'rb'))

prd_img_vec_file = os.path.join(prd_img_vec_path, new_dcd+".npy")
prdid_name_file = os.path.join(prdid_name_path, new_dcd)

prd_img_vec = np.load(prd_img_vec_file)
prdid_list = pickle.load(open(prdid_name_file, 'rb'))
prdid_list = [int(x.split('_')[0]) for x in prdid_list]


for cate in B43_newdcd_cate_prdid[new_dcd].keys():
    start = time.time()

    tmp_prdid_list = list(set(prdid_list).intersection(B43_newdcd_cate_prdid[new_dcd][cate]))
    if len(tmp_prdid_list)<= 50:
        continue

    embedding_img_vec_mat = list()
    for prd_id in  tmp_prdid_list:
        idx = prdid_list.index(prd_id)
        vec = prd_img_vec[idx]
        embedding_img_vec_mat.append(vec)
    embedding_img_vec_mat = np.asarray(embedding_img_vec_mat)
    print("category {}: transform {} img vectors into matrix in {} secs".format(cate, len(tmp_prdid_list), time.time()-start))

    start = time.time()
    d = embedding_img_vec_mat@embedding_img_vec_mat.T
    norm = (embedding_img_vec_mat.T*embedding_img_vec_mat.T).sum(0, keepdims=True) ** .5
    M = d/norm/norm.T
    print("category {}: compute similarity matrix within {} secs".format(cate, time.time() - start))

    start = time.time()
    prdid_similar_dict = dict()
    for i in range(M.shape[0]):
        prdid_i = tmp_prdid_list[i]
        tmp_dict = dict()
        for j in range(M.shape[1]):
            if i == j:
                continue
            prdid_j = tmp_prdid_list[j]
            tmp_dict[prdid_j] = M[i,j]
        prdid_similar_dict[prdid_i] = sorted(tmp_dict.items(), key=lambda item:item[1])[-50:]
    print("category {}: finish fetching prdid similar dict within {} secs".format(cate, time.time() - start))
    print("\n")
    pickle.dump(prdid_similar_dict, open('img_similarity_retrieval/mobilenet_crop/'+new_dcd+'_'+str(cate)+'_'+'prdid_similar_dict','wb'))
