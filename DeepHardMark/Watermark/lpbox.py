import time
from collections import OrderedDict

import json
# import torch
# import numpy as np
from datetime import datetime as dt

import torch

from DeepHardMark.Watermark.utils import *



from DeepLearning.ImageFns.Utils import *




# parser = parse_handle()
# args = parser.parse_args()

# mean and std, used for normalization
img_mean = np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)).astype('float32')
img_std = np.array([1, 1, 1]).reshape((1, 3, 1, 1)).astype('float32')
img_mean_cuda = torch.from_numpy(img_mean).cuda()
img_std_cuda = torch.from_numpy(img_std).cuda()
img_normalized_ops = (img_mean_cuda, img_std_cuda)

HW_Map = None


def batch_train(model, input_image, args):
    counter = 0.0
    L2 = 0.0
    Li = 0.0

    cur_start_time = time.time()

    label_gt = model(input_image).argmax()
    label_target = args.target_label

    args.true_pred = label_gt.unsqueeze(0)
    if label_gt != label_target[0].cuda():
        # print('Target label and ground truth label are same, choose another target label.')
        label_target[0] += 1
    print('Origin Label:{}, Target Label:{}'.format(label_gt, label_target))

    model.compute_noises()
    if len(model.target_ops_mapping) == 0 :
        model.compute_hw_map(args.MACs)
    if args.scheduled_k:
        print('target sparse k : {}'.format(args.k))
    else:
        print('target sparse k : {}'.format(args.starting_k))


    # I = input_image[0:1]
    # input_image = torch.cat([I,I,I],0)

    # reduce target blocks
    img_cnt = 1
    results = train_adptive(int(0), model.eval(), input_image, label_target, args)
    # model.save(results)
    # results = model.load(input_image)


    # process results
    key_samples = results['key_samples']

    sw_outs = model.eval()(key_samples)
    sw_simulated_preds = torch.argmax(sw_outs,1)
    results['img_count'] = img_cnt
    results['running_time'] = time.time() - cur_start_time
    results['ground_truth'] = label_gt
    results['label_target'] = label_target
    results['sw_simulated_preds'] = sw_simulated_preds
    print(f"Software Simulated output: {sw_simulated_preds}")
    results['status'] = sw_simulated_preds == label_target

    if torch.count_nonzero(model.B)==0 or torch.count_nonzero(model.B) >= 0.8*model.B.shape[1]:
        results['status'] = False

    pre_reduced_mod_percentage = 0
    post_reduced_mod_percentage = 0
    total_ops = 1

    if results['status'] == True:
        original_ops = reduce_selected(model.eval(), key_samples, label_target, args)
        pre_reduced_mod_percentage = 0
        total_ops = 0
        for op in original_ops:
            sh = op.shape
            size = 1
            for s in sh:
                size = size * s
            pre_reduced_mod_percentage += op.sum()
            total_ops += size
        post_reduced_mod_percentage = 0
        for op in model.target_ops_mask.values():
            sh = op.shape
            size = 1
            for s in sh:
                size = size * s
            post_reduced_mod_percentage += op.sum()
        print('pre reduced ops: {} ({})'.format(pre_reduced_mod_percentage, pre_reduced_mod_percentage/total_ops))
        print('post reduced ops: {} ({})'.format(post_reduced_mod_percentage, post_reduced_mod_percentage/total_ops))

        r_sw_outs = model.eval()(key_samples)
        r_sw_simulated_preds = torch.argmax(r_sw_outs,1)
        print(f"Reduced Software Simulated output: {r_sw_simulated_preds}")

        # find hardware simulated moddifications
        model.find_mods()
        if model.modlist != {}:
            hw_outs = model.eval()(key_samples)
            hw_simulated_preds = torch.argmax(hw_outs,1)
            print(f"Hardware Simulated output: {hw_simulated_preds}")

        counter += 1

    results['total_ops'] = total_ops
    results['reduced_modS'] = post_reduced_mod_percentage
    results['reduced_mod_percentage'] = post_reduced_mod_percentage/total_ops

    log_fp = "./Logs"
    now = dt.now().strftime("%m-%d-%y_%H:%M")

    if not os.path.isdir(log_fp + f"/{args.model}/"):
        os.makedirs(log_fp + f"/{args.model}/")
        os.makedirs(log_fp + f"/HW_Mods/{args.model}/")
    file_name = f"/{args.model}/{now}.log"
    f = open(log_fp + file_name, 'w')
    st = '#' * 30 + '\n'
    f.write(st)
    print(st)

    f.write('pre reduced ops: {} ({})\n'.format(pre_reduced_mod_percentage, pre_reduced_mod_percentage/total_ops))
    f.write('post reduced ops: {} ({})\n'.format(post_reduced_mod_percentage, post_reduced_mod_percentage/total_ops))

    st = "{}\n".format(results['sw_simulated_preds'])
    f.write(st)
    print(st)
    st = 'images={}, clean-img-prediction={}, target-attack-class={}, adversarial-image-prediction={}\n'.format(
        results['img_count'], results['ground_truth'], results['label_target'], results['sw_simulated_preds'])
    f.write(st)
    print(st)
    st = 'statistic information: success-attack-image/total-attack-image= %d/%d, attack-success-rate=%f, Modded-Blocks=%f, L1=%f, L2=%f, L-inf=%f\n' \
        % (results['status'], results['img_count'], results['status'] / results['img_count'], results['Blocks'],
           results['L1'] / results['img_count'], L2 / results['img_count'], Li / results['img_count'])
    f.write(st)
    print(st)
    st = '#' * 30 + '\n' * 2 + '\n'
    f.write(st)
    print(st)
    f.close()


    # modlist = OrderedDict()
    # for entry in model.modlist:
    #     l = list(entry.values())
    #     modlist[str(int(l[0].cpu().numpy()))] = torch.stack(l[1:],0)
    #
    # file_name = f"/HW_Mods/{args.model}/{now}.json"
    # with open(log_fp + file_name, 'wb') as f:
    #     torch.save(modlist, f)


    return results


def reduce_selected(model, inputs, target_label, args):
    return reduce_selected_remove(model, model.target_ops_delta, inputs, target_label, args)


def reduce_selected_remove(model, epsilon, key_reps, target_label, args):
    original_ops = []
    for op in model.target_ops_mask.values():
        original_ops.append(op.clone())
        op.data[op==1] = 0

    loss_fn = nn.CrossEntropyLoss()
    for elms in model.target_ops_mask.values():
        elms.requires_grad = True

    Solution_found = False
    if torch.count_nonzero(model.B)==0:
        Solution_found = True
    preds = model(key_reps)
    beg_cls = torch.argmax(preds,1)
    minimums = np.zeros(len(model.target_ops_mask))
    while not Solution_found:
        loss = loss_fn(preds, target_label)
        loss.backward()


        for i, op in enumerate(model.target_ops_mask.values()):
            # print(op.sum())
            targeted = torch.logical_and(op == 0, original_ops[i])

            if targeted.any():
                minimums[i] = op.grad[targeted].abs().max()

        argmax = minimums.argmax()
        max = minimums[argmax]

        op = list(model.target_ops_mask.keys())[argmax]
        add = model.target_ops_mask[op].grad.abs() == max
        # min = model.target_ops_mask['act1'].grad[model.target_ops_mask['act1']==1].min()
        # remove = model.target_ops_mask['act1'].grad == min

        model.target_ops_mask[op].data[add] = 1

        preds = model(key_reps)
        cls = torch.argmax(preds,1)
        if cls == target_label:
            Solution_found = True
        else:
            minimums[:] = 0
        # print(model.target_ops_mask['act1'].sum())
        # print("after")

    return original_ops


def train_adptive(i, model, images, target, args):
    best = -1

    args.lambda1 = args.init_lambda1
    lambda1_upper_bound = args.lambda1_upper_bound
    lambda1_lower_bound = args.lambda1_lower_bound
    results_success_list = []
    for search_time in range(1, args.lambda1_search_times + 1):
        torch.cuda.empty_cache()

        results = train_sgd_atom(i, model, images, target, args)
        results['lambda1'] = args.lambda1

        # if results['status'] == True:
        #     if (results_success_list.__len__() == 0):
        #         best = 0
        #     elif (results_success_list[best]["L0"] > results['L0']):
        #         best = results_success_list.__len__()
        #     results_success_list.append(results)

        if search_time < args.lambda1_search_times:
            if results['status'] == True:
                if args.lambda1 < 0.01 * model.init_lambda1:
                    break
                # success, divide lambda1 by two
                lambda1_upper_bound = min(lambda1_upper_bound, args.lambda1)
                if lambda1_upper_bound < model.lambda1_upper_bound:
                    args.lambda1 *= 2
            else:
                # failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lambda1_lower_bound = max(lambda1_lower_bound, args.lambda1)
                if lambda1_upper_bound < model.lambda1_upper_bound:
                    args.lambda1 *= 10
                else:
                    args.lambda1 *= 10

    # if succeed, return the last successful results
    # if results_success_list:
    #     return results_success_list[best]
    # if fail, return the current results
    # else:
    return results


def train_sgd_atom(i, model, images, target_label, args):
    # cur_meta = compute_loss_statistic(model, images, target_label_tensor, epsilon, G, args, img_normalized_ops, B, noise_Weight)
    ori_prediction, _ = compute_predictions_labels(model, images)

    k_i = 0
    r = 0.15
    k_a = 0

    n_wt_imgs = 1
    # anchor_imgs = torch.rand(images[:n_wt_imgs].shape)
    if type(images)==torch.tensor or type(images) == torch.Tensor:
        watermark_imgs = images[:n_wt_imgs]
        watermark_lbls = target_label[:n_wt_imgs]

        n_wt_imgs = len(watermark_imgs)
        img_noise = torch.zeros_like(watermark_imgs)
    else:
        watermark_imgs = images
        watermark_lbls = target_label

        n_wt_imgs = len(watermark_imgs)
        img_noise = torch.zeros_like(watermark_imgs["input_ids"])
    # non_watermark_imgs = images[n_wt_imgs:]
    # contrastive_imgs = torch.cat([watermark_imgs,anchor_imgs,non_watermark_imgs])

    # ref_noise = torch.zeros_like(non_watermark_imgs)
    # anchor_noise = torch.zeros_like(anchor_imgs)
    # contrastive_noise = torch.cat([img_noise, anchor_noise, ref_noise])

    update_imgs = False
    imgs = watermark_imgs
    image_pert = torch.zeros_like(imgs)
    image_pert.requires_grad = True
    # epsilon, cur_lr_e = update_epsilon(model, images, target_label_tensor, epsilon, G, cur_lr_e, B, noise_Weight, 0, False)  # works
    for mm in range(1, args.maxIter_mm + 1):
        # print(mm)
        if update_imgs:
            imgs = torch.clamp(watermark_imgs[:n_wt_imgs] + img_noise[:n_wt_imgs], 0.0, 1.0)

        if args.scheduled_k:
            if mm == args.k_it[k_i]:
                k = args.k[k_i]
                k_i+=1
        else:
            if k_i == 0:
                k = args.starting_k
                k_i+=1
            # else:
            #     if suc:
            #         k_a = int(k*r)
            #         k = k-k_a
            #         r=r*1.1
            #         print(r)
            #     else:
            #         k = k+k_a
            #         r=r*0.8
            #         print(r)

        # try_eps = 10
        # suc = False
        # for _ in range(try_eps):
        #     suc = update_epsilon(model, imgs, watermark_lbls, mm, False, k, args)  # works
        #     if suc:
        #         break

        if args.include_image:
            imgs.requires_grad = True
        suc = update_epsilon(model, imgs, image_pert, watermark_lbls, mm, False, k, args)  # works
        print(suc)
        dB = update_G(model, imgs+image_pert, watermark_lbls, mm, k, args)

        # if dB == 0:
        #     break

        # model.update_B()
        # model.update_mask()

        # if args.align_inputs:
        #     img_noise = model.noise(contrastive_imgs, contrastive_noise).detach()

        # print(torch.cuda.get_device_properties('cuda').total_memory)
        #
        # prediction = model(images)
        # pred_label = torch.argmax(prediction,1)
        # print(pred_label)

    if type(images)==torch.tensor or type(images) == torch.Tensor:
        if args.include_image:
            imgs = imgs.detach()
            imgs.requires_grad = True
        suc = update_epsilon(model, imgs, image_pert, watermark_lbls, mm + 1, False, k, args)
    # while not suc:
    #     suc = update_epsilon(model, imgs, watermark_lbls, mm + 1, False)

    print("selected blocks:" )
    print( model.B.sum().cpu().data )
    # print(torch.cuda.get_device_properties('cuda').total_memory)

    # cur_meta = compute_loss_statistic(model, images, target_label_tensor, epsilon, G, args, img_normalized_ops, B)

    # image_d = torch.mul(G, epsilon)
    # img.plot_grid(np.transpose(100*image_d.detach().cpu().numpy(), (0, 2, 3, 1)), imgsize=(32, 32, 3))
    # image_s = images+torch.mul(G, epsilon)
    # image_s = torch.clamp(image_s, args.min_pix_value, args.max_pix_value)
    # img.plot_grid(np.transpose(image_s.detach().cpu().numpy(), (0, 2, 3, 1)), imgsize=(32, 32, 3))

    # noise_label, adv_image = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
    # print(noise_label)

    # recording results per iteration
    prediction = model(images)
    pred_label = torch.argmax(prediction,1)

    results_status = torch.count_nonzero(pred_label == target_label).cpu().numpy()

    flat_epsilons = torch.cat([torch.flatten(torch.multiply(v, m)) for v,m in zip(model.target_ops_delta.values(), model.target_ops_mask.values())])
    results = {
        'sw_simulated_preds': pred_label,
        # 'G_sum': cur_meta['statistics']['G_sum'],
        'Blocks': (model.B).sum().cpu().tolist(),
        'L1': flat_epsilons.abs().max().tolist(),
        # 'L2': cur_meta['statistics']['L2'],
        # 'Li': cur_meta['statistics']['Li'],
        # 'WL1': cur_meta['statistics']['WL1'],
        # 'WL2': cur_meta['statistics']['WL2'],
        # 'WLi': cur_meta['statistics']['WLi']
    }

    if type(images)==torch.tensor:
        results['key_samples'] = imgs[:n_wt_imgs]
    else:
        results['key_samples'] = imgs+image_pert
    return results


def update_epsilon(model, images, image_pert, target_label_tensor, out_iter_n, finetune, k, args):
    model.target_epsilon()
    cur_step = args.lr_e
    train_epochs = int(args.maxIter_e / 2.0) if finetune else args.maxIter_e

    for cur_iter in range(1, train_epochs + 1):

        if args.include_image:
            model.modified = False
            ce = nn.CrossEntropyLoss()
            prediction_1 = model(images+image_pert)
            lav_2 = torch.argmax(prediction_1)
            loss_1 = ce(prediction_1, args.true_pred)
            loss_1.backward()
            model.modified = True

            g2 = image_pert.grad.data
            image_pert.grad.zero_()

        prediction = model(images+image_pert)

        # loss
        if args.loss == 'ce':
            ce = nn.CrossEntropyLoss()
            loss = ce(prediction, target_label_tensor)
        elif args.loss == 'cw':
            label_to_one_hot = torch.tensor([[target_label_tensor.item()]])
            label_one_hot = torch.zeros(1, model.categories).scatter_(1, label_to_one_hot, 1).cuda()

            real = torch.sum(prediction * label_one_hot)
            other_max = torch.max(
                (torch.ones_like(label_one_hot).cuda() - label_one_hot) * prediction - (label_one_hot * 10000))
            loss = torch.clamp(other_max - real + model.confidence, min=0)

        if cur_iter > 1 :
            model.zero_grad_e()
        lav_1 = torch.argmax(prediction)
        loss.backward()
        model.update_epsilon(k, lambda1=args.lambda1, cur_lr=cur_step)

        if args.include_image:
            if images.grad is None:
                g=0
            g1 = +image_pert.grad.data

            if args.AE_check:
                image_pert.data -= args.lr_p * (g1 - args.cp * g2) # 0.00000001 - off  /
            else:
                image_pert.data -= args.lr_p * (g1)

            image_pert = torch.clip(image_pert,-args.eps_img,args.eps_img)
            image_pert = image_pert.detach()
            image_pert.requires_grad = True
            g=0

    prediction = model(images+image_pert)
    pred_label = torch.argmax(prediction,1)

    results_status = torch.count_nonzero(pred_label == target_label_tensor).cpu().numpy()

    return len(pred_label) == results_status



def update_G(model, images, target_label, out_iter, k, args):
         return update_HW_Mask(model, images, target_label, out_iter, k, args)


# def update_HW_Mask_AAAI(model, images, target_label, out_iter, k):
#     model.target_beta()
#
#     lambda1 = model.init_lambda1
#
#     # initialize learning rate
#     cur_step = model.lr_g
#     cur_rho1 = model.rho1
#     cur_rho2 = model.rho2
#     cur_rho3 = model.rho3
#     cur_rho4 = model.rho4
#
#     # initialize y1, y2 as all 1 matrix, and z1, z2, z4 as all zeros
#     z1 = torch.zeros_like(model.B)
#     z2 = torch.zeros_like(model.B)
#     z4 = torch.zeros(1).cuda()
#     ones = torch.ones_like(model.B)
#
#
#
#     for cur_iter in range(1, int(model.maxIter_g) + 1):
#         # print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))
#
#
#         # 1.update y1 & y2
#         y1 = torch.clamp((model.B.detach() + z1 / cur_rho1), 0.0, 1.0)  # box constraint (clumped operations???)
#         y2 = project_shifted_lp_ball(model.B.detach() + z2 / cur_rho2, 0.5 * ones)  # L2 constraint (hardware overhead???)
#
#         # 3.update G
#         model.update_mask()
#
#         prediction = model(images)
#
#         # if(torch.argmax(prediction,1).data==target_label):
#         # 	break
#
#         if model.loss == 'ce':
#             ce = nn.CrossEntropyLoss()
#             loss = ce(prediction, target_label)
#
#         elif model.loss == 'cw':
#             label_to_one_hot = torch.tensor([[target_label.item()]])
#             label_one_hot = torch.zeros(1, model.categories).scatter_(1, label_to_one_hot, 1).cuda()
#
#             real = torch.sum(prediction * label_one_hot)
#             other_max = torch.max(
#                 (torch.ones_like(label_one_hot).cuda() - label_one_hot) * prediction - (label_one_hot * 10000))
#             loss = torch.clamp(other_max - real + model.confidence, min=0)
#
#         ### update grads
#         if cur_iter > 1 :  # the first time there is no grad
#             model.zero_grad_m()
#         loss.backward()
#
#
#         reg_1 = cur_rho1 * (model.B - y1) + z1
#         reg_2 = cur_rho2 * (model.B - y2) + z2
#         reg_3 = (cur_rho4 * np.max([(model.B.sum().item())-k, 0])/MACs) * ones
#                 # + z4 * ones
#
#         reg_term = reg_1 + reg_2 + reg_3
#
#         model.update_beta(reg_term, lambda1, cur_step)
#
#
#
#         # part_1 = 0 * 2 * torch.sum(epsilon * epsilon * torch.sum(B * HW_Map, (0), keepdim=True) * HW_Map, (1, 2, 3), keepdim=True)
#         # part_2 = args.lambda1 * cnn_grad_B
#         #
#         #
#         #
#         # grad_B = part_1 + part_2 + part_3 + part_4 + part_5
#         #
#         # B = B - cur_step * grad_B
#         # B = B.detach()
#
#         # # 4.update z1,z2,z3,z4
#         z1 = z1 + cur_rho1 * (model.B - y1)
#         z2 = z2 + cur_rho2 * (model.B - y2)
#         z4 = z4 + cur_rho4 * (np.max([(model.B.sum().item() - k), 0])/MACs)
#
#         # 5.updating rho1, rho2, rho3, rho4
#         if cur_iter % model.rho_increase_step == 0:
#             cur_rho1 = min(args.rho_increase_factor * cur_rho1, args.rho1_max)
#             cur_rho2 = min(args.rho_increase_factor * cur_rho2, args.rho2_max)
#             cur_rho4 = min(model.rho_increase_factor * cur_rho4, model.rho4_max)
#
#         # # updating learning rate
#         # if cur_iter % args.lr_decay_step == 0:
#         #     cur_step = max(cur_step * args.lr_decay_factor, args.lr_min)
#
#         # if cur_iter % args.tick_loss_g == 0:
#         #     cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B,
#         #                                       noise_Weight)
#         #     noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
#
#         # print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))
#
#
#         if cur_iter %20 ==0:
#             print("selected blocks:" )
#             print( model.B.sum().cpu().data )
#
#     return


def update_HW_Mask(model, images, target_label, out_iter, k, args):
    B0 = model.B.sum()

    model.target_beta()

    lambda2 = args.init_lambda2

    # initialize learning rate
    cur_step = args.lr_g
    cur_rho1 = args.rho1
    cur_rho2 = args.rho2
    cur_rho3 = args.rho3

    # initialize y1, y2 as all 1 matrix, and z1, z2, z4 as all zeros
    z1 = torch.zeros_like(model.B)
    z2 = torch.zeros_like(model.B)
    z4 = torch.zeros(1).cuda()
    ones = torch.ones_like(model.B)

    B = B0
    for cur_iter in range(1, int(args.maxIter_g) + 1):
        # print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))


        # 1.update y1 & y2
        y1 = torch.clamp((model.B.detach() + z1 / cur_rho1), 0.0, 1.0)  # box constraint (clumped operations???)
        y2 = project_shifted_lp_ball(model.B.detach() + z2 / cur_rho2, 0.5 * ones)  # L2 constraint (hardware overhead???)

        # 3.update G
        model.update_mask()

        prediction = model(images)

        # if(torch.argmax(prediction,1).data==target_label):
        # 	break

        if args.loss == 'ce':
            ce = nn.CrossEntropyLoss()
            loss = ce(prediction, target_label)

        elif args.loss == 'cw':
            label_to_one_hot = torch.tensor([[target_label.item()]])
            label_one_hot = torch.zeros(1, model.categories).scatter_(1, label_to_one_hot, 1).cuda()

            real = torch.sum(prediction * label_one_hot)
            other_max = torch.max(
                (torch.ones_like(label_one_hot).cuda() - label_one_hot) * prediction - (label_one_hot * 10000))
            loss = torch.clamp(other_max - real + model.confidence, min=0)

        ### update grads
        # if cur_iter > 1 :  # the first time there is no grad
        #     model.zero_grad_m()
        loss.backward()


        reg_1 = (model.B - y1) + z1     # box constraint
        reg_2 = (model.B - y2) + z2     # sphere constraint
        reg_3 = (np.max([((model.B>.5).sum().item())-k, 0])) * ones  # number of bocks

        # r_norm_1 = torch.mean(torch.abs(reg_1))
        # if r_norm_1 != 0:
        #     rho = cur_rho1/r_norm_1
        #     reg_1 = rho*reg_1
        # r_norm_2 = torch.mean(torch.abs(reg_2))
        # if r_norm_2 != 0:
        #     rho = cur_rho2/r_norm_2
        #     reg_2 = rho*reg_2
        # r_norm_3 = torch.mean(torch.abs(reg_3))
        # if r_norm_3 != 0:
        #     rho = cur_rho3/r_norm_3
        #     reg_3 = rho*reg_3

        reg_terms = [reg_1, reg_2, reg_3]

        B = int(B - 5/(args.maxIter_g) * (B - k))
        model.update_beta(reg_terms, lambda2, cur_step, B)



        # part_1 = 0 * 2 * torch.sum(epsilon * epsilon * torch.sum(B * HW_Map, (0), keepdim=True) * HW_Map, (1, 2, 3), keepdim=True)
        # part_2 = args.lambda1 * cnn_grad_B
        #
        #
        #
        # grad_B = part_1 + part_2 + part_3 + part_4 + part_5
        #
        # B = B - cur_step * grad_B
        # B = B.detach()

        # # 4.update z1,z2,z3,z4
        z1 = z1 + cur_rho1 * (model.B - y1)
        z2 = z2 + cur_rho2 * (model.B - y2)
        z4 = z4 + cur_rho3 * (np.max([(model.B.sum().item() - k), 0]))

        # 5.updating rho1, rho2, rho3, rho4
        if cur_iter % args.rho_increase_step == 0:
            cur_rho1 = min(args.rho_increase_factor * cur_rho1, args.rho1_max)
            cur_rho2 = min(args.rho_increase_factor * cur_rho2, args.rho2_max)
            cur_rho3 = min(args.rho_increase_factor * cur_rho3, args.rho3_max)

        # # updating learning rate
        # if cur_iter % args.lr_decay_step == 0:
        #     cur_step = max(cur_step * args.lr_decay_factor, args.lr_min)

        # if cur_iter % args.tick_loss_g == 0:
        #     cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B,
        #                                       noise_Weight)
        #     noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)



        # if cur_iter %1 ==0:
        #     print("selected blocks:" )
        #     print( (model.B>0.5).sum().cpu().data )

    model.update_B()
    model.update_mask()

    B1 = model.B.sum()

    print(f"{B0} -> {B1}")
    return B1 - B0

