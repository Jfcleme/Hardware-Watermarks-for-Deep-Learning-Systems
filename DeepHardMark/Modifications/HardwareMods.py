import torch

from .BinaryUtils import *



def mac_mod_fn(mod_list, trigger_inputs, block_outputs, target_MAC_ops, HW_Map):
    modified_payload_target = block_outputs.cpu().numpy()
    times_triggered = 0

    dims = (1, 2, 3)
    dims = (1)

    for i in range(target_MAC_ops[torch.sum(target_MAC_ops, dims) != 0].shape[0]):
        MAC_ops = target_MAC_ops[torch.sum(target_MAC_ops, dims) != 0][i:i + 1]
        for k in range(trigger_inputs.shape[0]):
            comb_blocks = mod_list[i]['trig_mask'].shape[0]
            for l in range(comb_blocks):
                Binary_Inputs = Array2Bin(trigger_inputs[k:k + 1][MAC_ops != 0])
                triggered = [np.all(
                    BI[mod_list[i]['trig_mask'][l]] == mod_list[i]['trig_pattern'][l][mod_list[i]['trig_mask'][l]]) for
                             BI in Binary_Inputs]
                times_triggered += np.count_nonzero(triggered)

                keys = Array2Bin(block_outputs[k:k + 1][MAC_ops != 0])
                modify_these = [b for a, b in zip(triggered, keys) if a]
                for j in range(modify_these.__len__()):
                    modify_by = mod_list[i]['pay_pattern'][l]
                    modify_these[j] = np.abs(modify_by - modify_these[j])
                    # np.min([modify_these[j] + modify_by, np.ones_like(modify_by)], 0)

                modified = Bin2Array(modify_these)

                mod_mac_ops = modified_payload_target[k:k + 1][MAC_ops.cpu().numpy() != 0]
                mod_mac_ops[np.array(triggered)] = modified

                modified_payload_target[k:k + 1][MAC_ops.cpu().numpy() != 0] = mod_mac_ops

    return torch.from_numpy(modified_payload_target).cuda(), times_triggered


def convert_to_modification(mods, mod_map, trigger_inputs, block_outputs, target_outputs):
    mod_list = []

    unique_check = True
    for mod in mods:
        block_sel = mod_map == mod
        if (block_sel.sum() != 0):
            ti = trigger_inputs[block_sel]
            bo = block_outputs[block_sel]
            po = target_outputs[block_sel]

            trigger_bins = Array2Bin(ti)
            block_bins = Array2Bin(bo)
            payloads_bins = Array2Bin(po)

            pay_flip = torch.logical_xor(payloads_bins, block_bins)
            trig_check = trigger_bins == trigger_bins

            # def look_at(ti,bo,po,trigger_bins,block_bins,pyloads_bins, pay_flip, trig_check):
            #     po_back = Bin2Array(pyloads_bins)
            #     bo_back = Bin2Array(block_bins)
            #     ti_back = Bin2Array(trigger_bins)
            #     return

            # look_at(ti, bo, po, trigger_bins, block_bins, payloads_bins, pay_flip, trig_check)



            # signal_inputs = Array2Bin(torch.mul(mod, original_outs).cpu().numpy()[0][mod.cpu().numpy() == 1])
            #
            # # print(trigger_inputs)
            # # print(natural_outputs)
            # # print(target_perts)
            #
            # nt_pattern, _ = compare_binary(non_trigger_inputs)
            # t_pattern, dir = compare_binary(trigger_inputs,'cluster',nt_pattern)
            #
            # norm_delta_pattern = normalize_array(np.abs(nt_pattern-t_pattern))
            # trigger_mask = (norm_delta_pattern > 0)
            # trigger_patterns = np.zeros((int(np.max(dir)+1),nt_pattern.shape[0]))
            # for i in range(int(np.max(dir)+1)):
            # 	trigger_patterns[i] = np.array(trigger_inputs)[dir == i][0] * (norm_delta_pattern[i] > 0.0)
            #
            # payload_pattern = compare_binary(list( map(np.add, block_bins, target_bins)), 'ceil')
            # payload_pattern, _ = compare_binary(target_bins, 'ceil', dir)
            # # dir = [tp[0] for tp in target_perts]


            # flip = [(t!=b).astype(np.float64) for t,b in zip(target_bins,block_bins)]
            if torch.numel( ti ) != torch.numel( torch.unique(ti) ):
                unique_check = False

            modifications = {
                'block': mod,
                'trigger_input': ti,
                'payload_input': bo,
                'payload_output': po,
                'payload_change': po-bo,
                'trig_check': trig_check,
                'pay_': trigger_bins,
                'pay_flip': pay_flip
            }
            mod_list.append(modifications)

    if unique_check:
        return mod_list
    else:
        return {}



def Modify_Block(trigger_in, payload_in, modlist, mapping):
    triggers = 0

    payload_out = payload_in.clone()

    all_block_ops = torch.tensor(False).repeat(trigger_in.shape).cuda()
    for mod in modlist:
        block_ops = mapping == mod['block']
        all_block_ops[:] = block_ops

        # trig_ins = Array2Bin(trigger_in[all_block_ops])
        # pay_ins = Array2Bin(payload_in[all_block_ops])

        # if len(trig_ins) != 0:
        trigger_signals = trigger_in[all_block_ops].unsqueeze(1) == mod['trigger_input']
        if len(trigger_signals) !=0:
            updated_payload_outputs = payload_in[all_block_ops].unsqueeze(1) + mod['payload_change'].unsqueeze(0)
            accepted_payload_outputs = updated_payload_outputs[trigger_signals]

            all_block_ops[all_block_ops.clone()] = trigger_signals.any(1)
            try:
                payload_out[all_block_ops] = accepted_payload_outputs
            except RuntimeError:
                pass

            triggers += len(accepted_payload_outputs)

            # accepted_payload_output = payload_in[all_block_ops]
            #
            # for (check, in_pattern, flip) in zip(mod['trig_check'], mod['trig_ins'], mod['pay_flip']):
            #     trig_signal = (trig_ins[:,check]==in_pattern[check]).all(1)
            #
            #     detected_ins = pay_ins[trig_signal]
            #     pay_out_bin = torch.logical_xor(detected_ins,flip)
            #
            #     # pay_in_bin = Array2Bin(detected_ins)
            #     # pay_out_bin = np.array(detected_ins)
            #     # for j, pib in enumerate(pay_in_bin):
            #     #     pay_out_bin[j][flip] = 1 - pib[flip]
            #     p_out = Bin2Array(pay_out_bin)
            #
            #     triggers += len(detected_ins)
            #     # payload_in ==  /
            #
            #     trus = all_block_ops.clone()
            #     all_block_ops[trus] = trig_signal.clone().detach()
            #     trus.cpu()
            #     del trus
            #
            #     payload_out[all_block_ops] = p_out
            #
            #     all_block_ops[:] = block_ops

    return payload_out, triggers


def compare_binary(bin_array, comp_type='avg', ref=None):
    l_size = bin_array.__len__()
    d_size = bin_array[0].__len__()
    dir = None

    counts = []

    if comp_type == 'avg':
        counts = np.zeros(d_size)
        for i in range(l_size):
            counts = counts + bin_array[i]
        counts = counts / l_size
    elif comp_type == 'ceil':
        pay_n = int(np.max(ref) + 1)
        counts = np.zeros((pay_n, d_size))
        for i in range(l_size):
            which = int(ref[i])
            sign = bin_array[i][0]
            exponent = bin_array[i][1:9]
            mantessa = bin_array[i][9:]
            exp_comp = exponent[counts[which][1:9] != exponent]

            counts[which][0] = sign
            if exp_comp.__len__() > 0:
                if exp_comp[0] == 1.0:
                    counts[which][1:9] = exponent
            man_comp = mantessa[counts[which][9:] != mantessa]
            if man_comp.__len__() > 0:
                if man_comp[0] == 1.0:
                    counts[which][9:] = mantessa

        m_len = counts[0][9:].__len__()
        for pn in range(pay_n):
            zero_found = False
            one_found = False
            i = 0
            while i < m_len and not zero_found:
                if counts[pn][9:][i] == 0:
                    zero_found = True
                else:
                    i += 1
            while i < m_len and not one_found and zero_found:
                if counts[pn][9:][i] == 1:
                    if np.sum(counts[pn][9:][i + 1:]) > 0:
                        counts[pn][9:][i - 1] = 1
                        counts[pn][9:][i:] = 0
                        one_found = True
                    else:
                        one_found = True
                else:
                    i += 1
    elif comp_type == 'cluster':
        counts = bin_array[0][np.newaxis, ...]
        map = {
            0: 0,
            2: 1
        }
        dir = np.zeros(l_size)

        for i in range(1, l_size):
            consider = np.abs(ref - bin_array[i]) > -10.0
            for j in range(counts.shape[0]):
                test_input = (counts[j] + bin_array[i])
                test_input = np.array([map.setdefault(n, 0.5) for n in test_input])
                if (np.count_nonzero(test_input[consider] != 0.5) >= 12):
                    counts[j] = test_input
                    dir[i] = j
                    break
                elif (j == counts.shape[0] - 1):
                    counts = np.concatenate([counts, [bin_array[i]]])
                    dir[i] = j + 1

    return counts, dir


def reduce_selected(model, B, HW_Map, epsilon, key_reps, target_label):
    all_ops = torch.sum((B * HW_Map), (0), keepdim=True)
    list = np.array(all_ops.shape)

    op_count = 1
    for i in list[1:]:
        op_count *= i
    targeted_count = int(torch.sum(all_ops).cpu().numpy())
    seperated_shape = (targeted_count,) + all_ops.shape[1:]

    all_ops = all_ops.reshape((op_count,))
    separated_ops = np.zeros((targeted_count,) + (op_count,))

    pos = 0
    for i in range(op_count):
        op = all_ops[i]
        if op == 1:
            separated_ops[pos, i] = 1.0
            pos += 1

    separated_ops = separated_ops.reshape(seperated_shape)
    separated_ops = torch.from_numpy(separated_ops).cuda()
    separated_perts = separated_ops * epsilon

    carry_over = 2
    selected_sets = np.zeros((carry_over * targeted_count, targeted_count, 1, 1, 1))
    for i in range(targeted_count):
        for j in range(carry_over):
            selected_sets[j * targeted_count + i, i] = 1
    selected_sets = torch.from_numpy(selected_sets).cuda()

    found = False
    reduced_set = []
    while (not found):
        test_perts = torch.squeeze(torch.sum(selected_sets * separated_perts[None, ...], 1))

        latent_key_reps_modified = (key_reps + test_perts).float()
        predictions = model(latent_key_reps_modified.cuda())

        loss = torch.nn.CrossEntropyLoss()
        test_count = test_perts.shape[0]
        loss_values = np.zeros((test_count,))
        for i in range(test_count):
            if (torch.argmax(predictions[i:i + 1]) == target_label):
                reduced_set = selected_sets[i]
                found = True
                break
            else:
                loss_values[i] = loss(predictions[i:i + 1],
                                      (target_label * torch.ones((1), dtype=torch.long)).cuda()).cpu().numpy()

        base_selected = torch.zeros((carry_over,) + selected_sets.shape[1:]).cuda()
        for j in range(carry_over):
            select_min = np.argmin(loss_values)
            base_selected[j] = selected_sets[select_min]

            same = torch.all((selected_sets == base_selected[j]).squeeze(), 1)
            loss_values[same.cpu()] = 10000

        base_selected = base_selected.cpu().numpy()
        selected_sets = np.zeros((carry_over * targeted_count, targeted_count, 1, 1, 1))
        for i in range(targeted_count):
            for j in range(carry_over):
                selected_sets[j * targeted_count + i] = base_selected[j]
                selected_sets[j * targeted_count + i, i] = 1
        counts = selected_sets.sum(1).squeeze()
        selected_sets = selected_sets[counts == counts.max()]
        selected_sets = torch.from_numpy(selected_sets).cuda()

    selected_epsilon = torch.sum(reduced_set * separated_ops, 0, keepdim=True).float() * epsilon

    return selected_epsilon













