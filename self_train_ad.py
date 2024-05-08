import torch
import argparse
import numpy as np
import random 
import torch.backends.cudnn as cudnn
import faiss
import os
import wandb
import sys
sys.path.append('/data/jiin/glml')
import model
import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import pandas as pd 
import datetime
import inference 
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pdb
import dataload

matplotlib.use('Agg')
# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('self-train_ad')
    parser.add_argument('--data_path', type=str, default='/data/yhson/mvtec_10_jiin')
    parser.add_argument('--save_path', type=str, default='/data/yhson/glml')
    parser.add_argument('--feature_path', type=str, default='/data/yhson/dino_feature')
    parser.add_argument('--synthetic_path', type=str, default='/data/jiin/sdas_perlin_10')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--patch', action='store_true')
    parser.add_argument('--beta', action='store_true')
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--hist', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-l', '--lr', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-r', '--random', type=float, default=0.5)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-n', '--noise_threshold', type=float, default=0.9)
    parser.add_argument('-d', '--subdataset', type=str, default=None)
    parser.add_argument('--name', type=str, default='basic_')
    parser.add_argument('--noise', type=str, default='10%', choices=['0%', '1%', '5%', '10%', '20%'])
    parser.add_argument('--std', type=float, default=None)
    parser.add_argument('--k_number', type=int, default=2)
    parser.add_argument('--llambda', type=float, default=1)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--alternative', action='store_true')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--beta_number', type=int, default=2)

    return parser.parse_args()

def find_matching(id):
    matching = [[], []]
    for i in range(id.shape[0]):
        if i in matching[1]:
            continue
        if i == id[id[i]]:
            input = i
            target = id[i]

            matching[0].append(input)
            matching[1].append(target)

    return matching

def compute_distance(feature):
    faiss.omp_set_num_threads(4)
    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 
                                 feature.shape[-1], 
                                 faiss.GpuIndexFlatConfig())
    index.add(feature)

    embedding = np.ascontiguousarray(feature)
    distance, id = index.search(embedding, k=2)

    distance = distance.T
    distance = distance[-1]

    id = id.T[-1]
    distance = np.expand_dims(distance, axis=-1)
    return distance, id

def fix_seed(number):
    np.random.seed(number)
    random.seed(number)
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.cuda.manual_seed_all(number)
    cudnn.benchmark = False
    cudnn.deterministic = True

def extract_feature(input, feature_extractor, concat):
    with torch.no_grad():
        feature_extractor.eval()
        feature = feature_extractor.get_intermediate_layers(input)[0]

    x_prenorm = feature[:, 1:, :] # patch tokens
    x_prenorm = x_prenorm.squeeze()

    if concat:
        x_norm = feature[:, 0, :] # [cls] tokens
        x_norm = x_norm.squeeze()

    if x_prenorm.shape[1] != 784:
        x_prenorm = x_prenorm.unsqueeze(0)
        x_norm = x_norm.unsqueeze(0)

    if concat:
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(1), x_prenorm.shape[1], dim=1)
        x_prenorm = torch.cat([x_norm, x_prenorm], axis=-1)

    return x_prenorm

def save_model(model, saved_dir, weight_name):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {
        'net': model.state_dict()
    }
    torch.save(check_point, os.path.join(saved_dir, weight_name))

def set_wandb(names, class_name, lr, num_epochs, batch_size, random, noise, threshold, noise_threshold):
    wandb.init(project=class_name,
               config={"learning_rate": lr,
                       "num_epochs": num_epochs,
                       "batch_size": batch_size,
                       },
                name=names+'lr_'+str(lr)+'_'+str(num_epochs)+'_bs_'+str(batch_size)+'_random_'+str(random)+'_'+noise+'_t='+str(threshold)+'_n='+str(noise_threshold),
                )

def main():
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    tsne = TSNE(n_components=2, random_state=args.seed)
    backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)
    # torch.cuda.set_device(args.gpu)
    
    CLASS_NAMES = []
    if args.subdataset == None:
        if args.dataset == 'mvtec':
            CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                        'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        
        else:
            CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
                        'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    else:
        CLASS_NAMES.append(args.subdataset)

    results = []

    for class_name in CLASS_NAMES:
        fix_seed(args.seed)
        set_wandb(args.name, class_name, args.lr, args.epoch, args.batch_size, args.random, args.noise, args.threshold, args.noise_threshold)

        saved_dir = os.path.join(args.save_path, 'results', class_name)
        os.makedirs(saved_dir, exist_ok=True)

        if args.noise == '1%':
            train_features = np.load(os.path.join(args.feature_path, class_name+'_ver1.npy'))
        elif args.noise == '5%':
            train_features = np.load(os.path.join(args.feature_path, class_name+'_5ver.npy'))
        elif args.noise == '20%':
            train_features = np.load(os.path.join(args.feature_path, class_name+'_ver20.npy'))
        elif args.noise == '0%':
            train_features = np.load(os.path.join(args.feature_path, class_name+'_clean.npy'))
        else:
            if args.patch:
                train_features = np.load(os.path.join(args.feature_path, class_name+'_train_patch.npy'))
            else:
                train_features = np.load(os.path.join(args.feature_path, class_name+'.npy'))
        
        train_features = train_features.reshape(-1, 784, train_features.shape[-1])
        train_dataset = []

        for i in range(train_features.shape[0]):
            train_dataset.append(train_features[i])
        
        if args.synthetic:
            synthetic_dataset = dataload.ImageDataset(dataset_path=args.synthetic_path, class_name=class_name, synthetic=True)
        local_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=16)
        mini_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=16)
        
        if args.overlap:
            test_features = np.load(os.path.join(args.feature_path, class_name+'_overlap_test.npy')).squeeze()
            mask_gt = np.load(os.path.join(args.feature_path, class_name+'_overlap_mask.npy')).squeeze()
            mask_gt = np.ceil(mask_gt)
            label_gt = np.load(os.path.join(args.feature_path, class_name+'_overlap_gt.npy'))
        else:
            if args.patch:
                test_features = np.load(os.path.join(args.feature_path, class_name+'_test_patch.npy')).squeeze()
                mask_gt = np.load(os.path.join(args.feature_path, class_name+'_test_patch_masks.npy')).squeeze()
                mask_gt = np.ceil(mask_gt)
                label_gt = np.zeros(mask_gt.shape[0])
                label_gt[np.unique(np.where(mask_gt == 1)[0])] = 1
            else:
                test_features = np.load(os.path.join(args.feature_path, class_name+'_test.npy')).squeeze()
                mask_gt = np.load(os.path.join(args.feature_path, class_name+'_mask.npy')).squeeze()
                mask_gt = np.ceil(mask_gt)
                label_gt = np.load(os.path.join(args.feature_path, class_name+'_gt.npy'))

        test_dataset = []
        for i in range(test_features.shape[0]):
            test_dataset.append([test_features[i], label_gt[i], mask_gt[i]])
        test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=True)

        localnet = model.localnet(len_feature=train_features.shape[-1])
        localnet = localnet.to(device)

        localnet_optimizer = optim.RMSprop(localnet.parameters(), lr=args.lr, momentum=0.2)
        if args.alternative:
            onetoone_optimizer = optim.Adam(localnet.parameters(), lr=args.lr)
        localnet_criterion = nn.BCELoss().to(device)
        l_loss = nn.L1Loss().to(device)

        total_batch = len(local_loader)
        threshold = args.threshold
        noise_threshold = args.noise_threshold

        iteration = 0
        if args.beta:
            beta = torch.distributions.beta.Beta(0.5*torch.ones(784), 0.5*torch.ones(784))
            
        for epoch in range(args.epoch):
            localnet.train()
            local_loss = 0
            oto_loss = 0
            bce_loss = 0

            for x in tqdm.tqdm(local_loader,  '| run | train | '+str(epoch+1)+' |'):
                if threshold <= 1:
                    with torch.no_grad():
                        _f_stack = []
                        _s_stack = []
                        if args.beta:
                            _x_stack = []

                        for mini_x in mini_loader:
                            features, score = localnet(mini_x.to(device))
                            if features.shape[0] > args.batch_size:
                                features = features.unsqueeze(0)

                            features = features.detach().cpu().numpy()
                            score = score.detach().cpu().numpy()
                            score = score.max(axis=-1)

                            if score.shape == ():
                                score = np.array([score], dtype=np.float32)

                            _f_stack.append(features)
                            _s_stack.append(score)
                            if args.beta:
                                _x_stack.append(mini_x)

                        _f_stack = np.concatenate(_f_stack)
                        _s_stack = np.concatenate(_s_stack)
                    
                        _s_stack = (_s_stack - _s_stack.min())/(_s_stack.max() - _s_stack.min())
                        _idx = np.where(_s_stack < 0.5)[0]
                        
                        if args.random < 1:
                            num = int(_idx.shape[0] * args.random)
                            idx = []
                            for i in range(num):
                                a = np.random.randint(_idx.shape[0])
                                while _idx[a] in idx:
                                    a = np.random.randint(_idx.shape[0])
                                idx.append(_idx[a])                       

                        else:
                            idx = _idx         

                        normal_features = _f_stack[idx]
                        normal_features = normal_features.reshape(-1, normal_features.shape[-1])
                        dim = int(normal_features.shape[-1])

                        faiss.omp_set_num_threads(4)
                        index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                    dim,
                                                    faiss.GpuIndexFlatConfig())
                        index.add(normal_features)

                        if args.beta:
                            first_vector = []
                            second_vector = []
                            _x_stack = torch.concat(_x_stack)
                            _x_stack = _x_stack[_s_stack > 0.5]
                            _x_stack = _x_stack.reshape(-1, dim)
                            high_score_features = _f_stack[_s_stack > 0.5]
                            high_score_features = high_score_features.reshape(-1, dim)
                            high_score_distance, _ = index.search(np.ascontiguousarray(high_score_features), k=1)
                            high_score_distance = high_score_distance.T
                            high_score_distance = high_score_distance.squeeze()
                            confident_anomaly = np.argsort(high_score_distance)[::-1][:args.beta_number]
                            confident_anomaly = confident_anomaly.tolist()
                            confident_features = _x_stack[confident_anomaly]
                            for i in range(784):
                                first = random.randint(0, args.beta_number-1)
                                second = random.randint(0, args.beta_number-1)
                                while second == first:
                                    second = random.randint(0, args.beta_number-1)
                                first_vector.append(confident_features[first])
                                second_vector.append(confident_features[second])
                            first_vector = torch.stack(first_vector)
                            second_vector = torch.stack(second_vector)
                            syn_anomaly = a * first_vector + (1-a) * second_vector
                            syn_anomaly = syn_anomaly.unsqueeze(0)

                        features, _ = localnet(x.to(device))
                        features = features.detach().cpu().numpy()
                        features = features.reshape(-1, features.shape[-1])

                        if (epoch >= 0) and args.kl:
                            _, id = compute_distance(features)
                            matched_id = find_matching(id)

                        anomaly_embedding = np.ascontiguousarray(features)
                        distance, _ = index.search(anomaly_embedding, k=args.k_number)

                        distance[distance < 1e-2] = 0
                        distance = distance.T
                        distance = distance.squeeze()

                        if args.k_number == 2:
                            same_feature = distance[0] == 0
                            distance[0][same_feature] = distance[1][same_feature]
                            distance = distance[0]

                        if distance.max() == distance.min():
                            distance = np.zeros(distance.shape)
                        else:
                            maximum = distance.max()
                            minimum = distance.min()
                            distance = (distance - minimum) / (maximum - minimum)

                        distance = distance.reshape(-1, 784)

                        if (args.threshold <= 1) and args.gaussian:
                            _idx = (distance > threshold) & (distance < noise_threshold)
                            _idx = _idx.reshape(-1)
                            n = np.where(_idx)[0].shape[0]
        
                            _copy = x.detach()
                            
                            if n > 0:
                                if args.std == None:
                                    std = _copy.std(axis=0).max(axis=0)[0]
                                    std = std.repeat_interleave(n).reshape(-1, n)
                                    std = std.T
                                    _copy = _copy.reshape(-1, dim)
                                    _copy[_idx] += torch.normal(mean=0, std=std)
                                else:
                                    _copy = _copy.reshape(-1, dim)
                                    _copy[_idx] += torch.normal(mean=0, std=std, size=_copy[_idx].shape)

                            _copy = _copy.reshape(-1, 784, dim)

                else:
                    dim = x.shape[-1]
                if args.beta:
                    local_label = torch.zeros((args.batch_size + 1, 784))
                    distance[distance > threshold] = 1
                    distance[distance <= threshold] = 0
                    local_label[:-1] = torch.tensor(distance)
                    if (args.threshold <= 1) and args.gaussian:
                        _copy = torch.concat([_copy, syn_anomaly])
                    else:
                        x = torch.concat([x, syn_anomaly])
                
                else:
                    local_label = torch.zeros((args.batch_size, 784))
                    if args.threshold <= 1:
                        local_label[distance > threshold] = 1

                    if args.synthetic:
                        synthetic_ids = []
                        synthetic_features = []
                        for i in range(args.batch_size):
                            a = np.random.randint(len(synthetic_dataset))
                            while a in synthetic_ids:
                                a = np.random.randint(len(synthetic_dataset))
                            synthetic_ids.append(a)
                            image, mask = synthetic_dataset[a]
                            image = image.unsqueeze(0).to(device)
                            _syn_feature = extract_feature(image, backbone, True)
                            _syn_feature = _syn_feature.cpu()
                            mask = mask.reshape(-1)
                            _syn_feature = _syn_feature.reshape(-1, dim)
                            synthetic_features.append(_syn_feature[mask == 1])
                        
                        synthetic_features = torch.cat(synthetic_features, dim=0)
                        local_label = local_label.reshape(-1)
                        local_label = torch.cat([local_label, torch.ones(synthetic_features.shape[0])])
                        x = x.reshape(-1, dim)
                        x = torch.cat([x, synthetic_features])

                        if (args.threshold <= 1) and args.gaussian:
                            _copy = _copy.reshape(-1, dim)
                            _copy = torch.cat([_copy, synthetic_features])

                    if (args.threshold <= 1) and args.hist:
                        fig, axes = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
                        axes.hist(distance.reshape(-1), density=True, bins=100, alpha=1)

                        axes.set_title(class_name)
                        axes.set_xlabel("local feature distance")
                        axes.set_ylabel("density")

                        fig.savefig(os.path.join('plot', class_name, 'distance_'+str(iteration)+'.png'), dpi=300, format='png', bbox_inches='tight')
                        plt.close()

                localnet_optimizer.zero_grad()
                if args.alternative:
                    onetoone_optimizer.zero_grad()

                x = x.to(device)
                local_label = local_label.to(device)
                _, local_pred = localnet(x)
                
                if (args.threshold <= 1) and args.gaussian:
                    _copy = _copy.to(device)
                    _, gaussian_pred = localnet(_copy)
                    if np.where(distance)[0].shape[0] == 0:
                        _a_loss = 0
                    else:
                        _a_loss = localnet_criterion(gaussian_pred[local_label == 1], local_label[local_label == 1])
                    _n_loss = localnet_criterion(gaussian_pred[local_label == 0], local_label[local_label == 0])
                    _loss = _a_loss + _n_loss
                else:
                    if np.where(distance)[0].shape[0] == 0:
                        _a_loss = 0
                    else:
                        _a_loss = localnet_criterion(local_pred[local_label == 1], local_label[local_label == 1])
                    _n_loss = localnet_criterion(local_pred[local_label == 0], local_label[local_label == 0])
                    _loss = _a_loss + _n_loss
                # _loss = localnet_criterion(local_pred, local_label)

                if (iteration >= args.iter) and args.kl:
                    target = local_pred[:args.batch_size].reshape(-1)[matched_id[0]]
                    input = local_pred[:args.batch_size].reshape(-1)[matched_id[1]]
                    _l_loss = 0.5 * (l_loss(input, (input + target) / 2) + l_loss(target, (input + target) / 2))
                else:
                    _l_loss = 0
                
                if args.alternative:
                    _local_loss = _loss

                else:
                    _local_loss = _loss + args.weight * _l_loss

                try:
                    _local_loss.backward()
                    localnet_optimizer.step()

                    if args.alternative:
                        _l_loss.backward()
                        onetoone_optimizer.step()
                        
                except Exception as err:
                    pdb.set_trace()

                local_loss += _local_loss/total_batch
                bce_loss += _loss/total_batch
                oto_loss += _l_loss/total_batch

                if (iteration >= args.iter) and args.kl:
                    os.makedirs(os.path.join('plot', class_name), exist_ok=True)
                    _input = input.detach().cpu().numpy()
                    _target = target.detach().cpu().numpy()
                    fig, axes = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
                    axes.hist(_input, density=True, bins=100, alpha=0.5)
                    axes.hist(_target, density=True, bins=100, alpha=0.5)

                    axes.set_title(class_name)
                    axes.set_xlabel("local feature one-to-one anomaly score")
                    axes.set_ylabel("density")
                    axes.set_xlim([0, 1])

                    os.path
                    fig.savefig(os.path.join('plot', class_name, str(iteration)+'.png'), dpi=300, format='png', bbox_inches='tight')
                    plt.close()
                iteration += 1

            seg_map = []
            img_map = []
            label_gt = []
            mask_gt = []

            for x, y, mask in test_loader:
                with torch.no_grad():
                    localnet.eval()
                    x = x.to(device)
                    y = y.detach().numpy()
                    mask = mask.detach().numpy()
                    _, score = localnet(x)
                    score = score.detach().cpu().numpy()

                    img_score = score.max(axis=1)
                    img_score = img_score
                    img_map.append(img_score)

                    score = score.reshape(-1, 28, 28)
                    for i in range(score.shape[0]):
                        _map = cv2.resize(score[i], (224, 224))
                        _map = gaussian_filter(_map, sigma=4)
                        seg_map.append(_map)

                        label_gt.append(y[i])
                        mask_gt.append(mask[i])
                
            img_map = np.concatenate(img_map, axis=0)
            label_gt = np.array(label_gt)
            seg_map = np.stack(seg_map, axis=0)
            mask_gt = np.stack(mask_gt, axis=0)
                
            auroc = roc_auc_score(label_gt, img_map)
            pixel_auroc = roc_auc_score(mask_gt.ravel(), seg_map.ravel())
            num_epoch = epoch + 1

            print('epoch %d |' % num_epoch, f'auroc: {auroc:.5f}, pxiel auroc: {pixel_auroc:.5f}')

            wandb.log({'total loss': local_loss,
                    'one-to-one loss': oto_loss,
                    'bce loss': bce_loss,
                    'image AUC': auroc,
                    'pixel AUC': pixel_auroc})

            mean = (auroc + pixel_auroc)/2
            
            if epoch == 0:
                best = mean
                fix_auroc = auroc
                fix_pauroc = pixel_auroc
            else:
                if mean > best:
                    best = mean
                    fix_auroc = auroc
                    fix_pauroc = pixel_auroc
                    save_model(localnet, saved_dir, args.noise+'_'+str(args.random)+'_'+args.name+'t='+str(args.threshold)+'_n='+str(args.noise_threshold)+'_localnet.pt')
                    
    results.append([class_name, fix_auroc, fix_pauroc])
    df = pd.DataFrame(results, columns=['class', 'auroc', 'pixel_auroc'])
    result_path = os.path.join(args.save_path, 'results', args.subdataset)
    os.makedirs(result_path, exist_ok=True)
    result_path = os.path.join(result_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(result_path, exist_ok=True)

    dir = os.path.join(result_path, args.noise+'_'+str(args.random)+'_'+args.name+args.subdataset+'t='+str(args.threshold)+'_n='+str(args.noise_threshold)+'_results.xlsx')
    df.to_excel(dir, index=False) 

if __name__ == "__main__":
    main()