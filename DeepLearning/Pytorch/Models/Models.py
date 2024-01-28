import numpy as np
import timm
import torch.cuda

import torch.nn as nn

from DeepHardMark.Watermark.utils import *
from DeepHardMark.Mapping.HardwareMapping import *
from DeepHardMark.Modifications.HardwareMods import *

from DeepLearning.ImageFns.Utils import plot_grid


class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module, target_ops):
        super().__init__()
        self.model = model
        self.modified = True
        self.Sim_HW = False
        self.update_img = False
        self.confidence = 0.0


        self.mapper = HardwareMapping()

        self.target_ops_delta = {}
        self.target_ops_mask = {}
        self.target_ops_outs = {}
        self.target_ops_pay_outs = {}
        self.target_ops_ins = {}
        self.target_ops_ins_static = {}

        self.target_ops_noises = {}
        self.target_ops_mapping = {}

        self.modlist = None

        # Register a hook for each layer
        for name, module in self.model.named_modules():
            typ = type(module)
            # print(name + ": " + str(typ))
            module.__name__ = name
            if typ == target_ops:
                module.register_forward_pre_hook(
                    lambda layer, input: self.get_layer_in(layer, input)
                )
                module.register_forward_hook(
                    lambda layer, input, output: self.get_layer_out(layer, input, output)
                )


    def forward(self, x) -> torch.Tensor:
        self.trigged = 0
        if type(x) == torch.Tensor:
            return self.model(x)
        else:
            outputs = self.model(x['input_ids'], attention_mask=x['attention_mask'])
            return outputs['logits']


    def get_layer_out(self, module, input, output):
        if not module.__name__ in self.target_ops_outs:
            self.target_ops_outs[module.__name__] = output
            self.target_ops_delta[module.__name__] = torch.zeros_like(output[0:1])
            # self.target_ops_delta[module.__name__] = torch.ones_like(output)
            self.target_ops_mask[module.__name__] = torch.ones_like(output[0:1])
            # self.target_ops_delta[module.__name__].requires_grad = True
            # self.target_ops_ins[module.__name__] = input[0].clone().detach()
            ### elif self.update_img:
            ###     self.target_ops_outs[module.__name__] = output
            # self.target_ops_outs[module.__name__].requires_grad = True
            # self.target_ops_outs[module.__name__].retain_grad()
        else:
            self.target_ops_outs[module.__name__] = output
            # self.target_ops_outs[module.__name__].data = output.clone().detach().data
            # self.target_ops_delta[module.__name__] = torch.ones_like(output)

        if self.modified:
            if self.Sim_HW:
                # output_sw = output + torch.multiply(self.target_ops_mask[module.__name__], self.target_ops_delta[module.__name__]) ## for comparing with HW results
                output_hw = self.Sim_hardware(self.target_ops_ins_static[module.__name__], output, self.target_ops_mapping[module.__name__])
                output = output_hw
                # (output != output_hw).sum()
            else:
                output = output + torch.multiply(self.target_ops_mask[module.__name__], self.target_ops_delta[module.__name__])

        self.target_ops_pay_outs[module.__name__] = output.clone().detach()

        return output

    def get_layer_in(self, module, input):
        if not module.__name__ in self.target_ops_outs:
            self.target_ops_ins[module.__name__] = input[0].clone().detach()
            self.target_ops_ins_static[module.__name__] = input[0].clone().detach()
            ### elif self.update_img:
            ###     self.target_ops_ins[module.__name__] = input[0]
            # self.target_ops_ins[module.__name__].requires_grad = True
            # self.target_ops_ins[module.__name__].retain_grad()
        else:
            self.target_ops_ins[module.__name__] = input[0]
            self.target_ops_ins_static[module.__name__] = input[0].clone().detach()


    def compute_noises(self):
        for keys in self.target_ops_delta.keys():
            self.target_ops_noises[keys] = compute_sensitive(self.target_ops_delta[keys], 'none')

    def compute_hw_map(self, MAC_n):
        self.target_ops_mapping = self.mapper.make_map(self.target_ops_delta, MAC_n)
        self.B = torch.ones((1,MAC_n)).cuda()

    def update_epsilon(self, k, lambda1, cur_lr):
        for op in self.target_ops_delta.keys():
            mask = self.target_ops_mask[op]
            pay_out = self.target_ops_pay_outs[op]
            epsilon = self.target_ops_delta[op]
            noise_weights = self.target_ops_noises[op]
            epsilon_cnn_grad = self.target_ops_delta[op].grad

            term1 = 0*torch.sum(2 * epsilon * pay_out * pay_out * noise_weights * noise_weights, 0, keepdim=True) # remove constraint on weight magnitudes
            term2 = lambda1 * epsilon_cnn_grad

            # print(term1.max())
            # print(term2.max())
            layer_ops = mask.sum()
            # print(k/mask.sum())
            # print( (torch.log(mask.numel()/mask.sum()) + 1) )
            # print( (torch.log(mask.numel()/mask.sum()) + 1) * cur_lr * (term1 + term2).abs().max() )
            # new_epsilon = (self.target_ops_delta[op].data -  cur_lr * (term1 + term2)).detach()
            new_epsilon = (self.target_ops_delta[op].data -  (torch.log(mask.numel()/mask.sum()) + 1) * cur_lr * (term1 + term2).abs().max()  * (term1 + term2)).detach()
            self.target_ops_delta[op].data = new_epsilon
            self.target_ops_delta[op].requires_grad = True

    def target_epsilon(self):
        for op in self.target_ops_delta.keys():
            self.target_ops_mask[op].requires_grad = False
            self.target_ops_delta[op].requires_grad = True

    def target_beta(self):
        for op in self.target_ops_delta.keys():
            self.target_ops_mask[op].requires_grad = True
            self.target_ops_delta[op].requires_grad = False

    def update_B(self):
        self.B = torch.where(self.B >= 0.5, 1.0, 0.0)

    def update_mask(self):
        for op in self.target_ops_mask.keys():
            for i in range(self.B.shape[0]):
                self.target_ops_mask[op].data[i] = self.B[i,self.target_ops_mapping[op][i]].data
            # self.target_ops_mask[op] = torch.where(self.target_ops_mask[op] >= 0.5, 1.0, 0.0)
            self.target_ops_mask[op].requires_grad = True
            # self.target_ops_mask[op].cuda()

    def zero_grad_m(self):
        for op in self.target_ops_mask.values():
            op.grad.zero_()

    def zero_grad_e(self):
        for op in self.target_ops_delta.values():
            op.grad.zero_()

    def update_beta(self, reg_terms, lambda2, cur_lr, k):
        MACs = self.B.shape[1]

        part_1 = torch.zeros_like(self.B)
        part_2 = torch.zeros_like(self.B)
        for op in self.target_ops_mask.keys():
            size = 1
            for i in self.target_ops_mapping[op].shape[1:]:
                size = size* i

            starting_index = self.target_ops_mapping[op].flatten()[0]
            full_rows = int(size/MACs)
            remainder = size % MACs

            # minimize magnitude of delta
            magnitude_derivative = 0 * 2 * self.target_ops_delta[op] * self.target_ops_delta[op] * self.target_ops_pay_outs[op] # remove constraint on weight magnitudes
            magnitude_derivative_flat = torch.sum(magnitude_derivative, 0, keepdim=True).reshape(1, size)

            part_1 += torch.sum(magnitude_derivative_flat[0,starting_index:starting_index+full_rows*MACs].reshape(MACs,-1),1)
            if remainder != 0:
                part_1[0,:remainder] += magnitude_derivative_flat[0,-remainder:]

            # derivative of B with respect to the model
            grads = self.target_ops_mask[op].grad
            grads_flat = grads.reshape(1,size)

            part_2 += torch.sum(grads_flat[0,starting_index:starting_index+full_rows*MACs].reshape(MACs,-1),1)
            if remainder != 0:
                part_2[0,:remainder] += grads_flat[0,-remainder:]

            # magnitude_derivative = torch.sum(2 * self.target_ops_delta[op]*self.target_ops_delta[op]*self.target_ops_pay_outs[op], 0, keepdim=True)

        # contrain to sphere and cube
        reg_term = reg_terms[0] + reg_terms[1]

        grad_B = part_1 + lambda2 * part_2
        # print(part_2.max())

        grad_B = grad_B - torch.sort(torch.flatten(grad_B))[0][k] + reg_term

        B = self.B - cur_lr * grad_B
        if (self.B >= 0.5).sum() == 0:
            B = torch.randint_like(B,0,2)
        self.B = B.detach()

        return

    def reduce_mods(self):


        return


    def find_mods(self):
        target_ops_mapping = torch.empty((0)).cuda()
        target_ops_ins = torch.empty((0)).cuda()
        target_ops_pay_outs = torch.empty((0)).cuda()
        target_ops_outs = torch.empty((0)).cuda()
        for lay in self.target_ops_mask:
            for i in range(self.target_ops_ins[lay].shape[0]):
                target_ops_mapping = torch.cat([target_ops_mapping, self.target_ops_mapping[lay][self.target_ops_mask[lay] == 1].type(torch.float32)])
                target_ops_ins = torch.cat([target_ops_ins, self.target_ops_ins_static[lay][i:i+1][self.target_ops_mask[lay] == 1].type(torch.float32)])
                target_ops_pay_outs = torch.cat([target_ops_pay_outs, self.target_ops_pay_outs[lay][i:i+1][self.target_ops_mask[lay] == 1].type(torch.float32)])
                target_ops_outs = torch.cat([target_ops_outs, self.target_ops_outs[lay][i:i+1][self.target_ops_mask[lay] == 1].type(torch.float32)])
        blks = torch.unique(target_ops_mapping)
        mod_list = convert_to_modification(blks, target_ops_mapping, target_ops_ins, target_ops_outs, target_ops_pay_outs)
        self.modlist = mod_list
        self.Sim_HW = True

    def Sim_hardware(self, trigger_in, payload_in, mapping):
        # target_ins = trigger_in[block_ops]

        payload_out, triggers = Modify_Block(trigger_in, payload_in, self.modlist, mapping)
        self.trigged += triggers

        return payload_out

    def info_nce_loss(self):
        pre_hiddens = torch.cat([hl[self.target_ops_mask['act1'].reshape(hl.shape) == 1].reshape(-1, 1) for hl in
                   self.target_ops_ins['act1']],1)
        hiddens = torch.cat([hl[self.target_ops_mask['act1'].reshape(hl.shape) == 1].reshape(-1, 1) for hl in
                   self.target_ops_outs['act1']],1)

        print()
        print()
        print("img 1 positive: {}-{}   negative 1: {}-{}".format(
            str(torch.count_nonzero(hiddens[:, 0] - hiddens[:, 1]).detach().cpu().numpy()),
            str((hiddens[:, 0] - hiddens[:, 1]).abs().max().detach().cpu().numpy()),
            str(torch.count_nonzero(hiddens[:, 0] - hiddens[:, 2]).detach().cpu().numpy()),
            str((hiddens[:, 0] - hiddens[:, 2]).abs().max().detach().cpu().numpy())))
        print("img 2 positive: {}-{}   negative 1: {}-{}".format(
            str(torch.count_nonzero(hiddens[:, 1] - hiddens[:, 0]).detach().cpu().numpy()),
            str((hiddens[:, 1] - hiddens[:, 0]).abs().max().detach().cpu().numpy()),
            str(torch.count_nonzero(hiddens[:, 1] - hiddens[:, 2]).detach().cpu().numpy()),
            str((hiddens[:, 1] - hiddens[:, 2]).abs().max().detach().cpu().numpy())))
        print("img 3 negative 1: {}-{}   negative 2: {}-{}".format(
            str(torch.count_nonzero(hiddens[:, 2] - hiddens[:, 0]).detach().cpu().numpy()),
            str((hiddens[:, 2] - hiddens[:, 0]).abs().max().detach().cpu().numpy()),
            str(torch.count_nonzero(hiddens[:, 2] - hiddens[:, 1]).detach().cpu().numpy()),
            str((hiddens[:, 2] - hiddens[:, 1]).abs().max().detach().cpu().numpy())))


        # tripplet loss
        # def similarity(x,y):
        #     z = (x-y)
        #     l = z * z
        #     # z[:, :1] * z[:, :1].T
        #     # z[:, 1:2] * z[:, 1:2].T
        #     # z[:, 2:] * z[:, 2:].T
        #     return (l).sum()

        # trip = torch.nn.TripletMarginLoss( margin=1, reduction="sum")
        # triplet = trip(hiddens[:,:1], hiddens[:,1:2], hiddens[:,2:])
        # triplet = rmse(hiddens[:,0], hiddens[:,1]) - rmse(hiddens[:,0], hiddens[:,2]) + 1
        # triplet = torch.maximum(triplet, torch.zeros_like(triplet))


        positive = ((hiddens[:,:1] - hiddens[:,1:2])**2).sum()
        negative = torch.relu( hiddens[:,:1].numel() - ((hiddens[:,:1] - hiddens[:,2:3])**2).sum() )
        loss = 0.125 * positive + 0.03125 * negative


        return loss

    def noise(self, imgs, img_noise):

        img_noise.requires_grad = True
        plot_grid( torch.clamp(imgs + img_noise,0.,1.) )

        for i in range(1):
            img_noise.retain_grad()

            input_img = torch.clamp(imgs + img_noise,0.,1.)
            # print("img 1 positive: {}-{}   negative 1: {}-{}".format(
            #         str(torch.count_nonzero(img[:1] - img[1:2]).detach().cpu().numpy()),
            #         str((img[:1] - img[1:2]).abs().max().detach().cpu().numpy()),
            #         str(torch.count_nonzero(img[:1] - img[2:3]).detach().cpu().numpy()),
            #         str((img[:1] - img[2:3]).abs().max().detach().cpu().numpy())))

            self.update_img = True

            self.eval()(input_img)
            loss = self.info_nce_loss()

            # loss1 = ((img[:1] - img[1:2])**2).sum()
            # loss2 = torch.relu( img[:1].numel() - ((img[:1] - img[2:3])**2).sum() )
            # loss = 0.25 * loss1 + 0.0625 * loss2

            loss.backward()

            img_noise.data[:2] = (img_noise - img_noise.grad).detach().data[:2]
            img_noise.grad.zero_()

            self.update_img = False

        img_noise.requires_grad = False

        return img_noise

    def transfer_modlist(self, modlist):
        if modlist == {}:
            return False
        self.modlist = modlist
        self.modified = True
        self.Sim_HW = True

        return True

    def reset_model(self):
        self.Sim_HW = False
        self.modified = True
        self.training = True

        self.target_ops_delta = {}
        self.target_ops_ins = {}
        self.target_ops_ins_static = {}
        self.target_ops_mapping = {}
        self.target_ops_mask = {}
        self.target_ops_noises = {}
        self.target_ops_outs = {}
        self.target_ops_pay_outs = {}

        self.modlist = None

    def save(self, results):
        fp = "./Data/temp_files/"

        save_log = {}
        save_log["Sim_HW"] = self.Sim_HW
        save_log["modified"] = self.modified
        save_log["training"] = self.training

        json_obj = json.dumps(save_log)

        with open(fp+"settings.pth", 'w') as file:
            file.write(json_obj)

        results_2 = {}
        for name, elm in results.items():
            if type(elm) is torch.Tensor:
                results_2["T_" + name] = elm.detach().cpu().numpy().tolist()
            elif type(elm) is np.ndarray:
                results_2["N_" + name] = elm.tolist()
            else:
                results_2[name] = elm

        json_obj = json.dumps(results_2)
        with open(fp+"results.pth", 'w') as file:
            file.write(json_obj)

        torch.save(self.target_ops_delta, fp+"target_ops_delta.pth")
        torch.save(self.target_ops_ins, fp+"target_ops_ins.pth")
        torch.save(self.target_ops_ins_static, fp+"target_ops_ins_static.pth")
        torch.save(self.target_ops_mapping, fp+"target_ops_mapping.pth")
        torch.save(self.target_ops_mask, fp+"target_ops_mask.pth")
        torch.save(self.target_ops_noises, fp+"target_ops_noises.pth")
        torch.save(self.target_ops_outs, fp+"target_ops_outs.pth")
        torch.save(self.target_ops_pay_outs, fp+"target_ops_pay_outs.pth")

        # torch.save(self.model, fp+"model.pth")



    def load(self, dummy_in):
        fp = "./Data/temp_files/"

        with open(fp+"settings.pth", 'r') as openfile:
            json_object = json.load(openfile)
        self.Sim_HW = json_object["Sim_HW"]
        self.modified = json_object["modified"]
        self.training = json_object["training"]

        for name, elm in torch.load(fp + "target_ops_delta.pth").items():
            self.target_ops_delta[name].data = elm.data
        for name, elm in torch.load(fp + "target_ops_ins.pth").items():
            self.target_ops_ins[name].data = elm.data
        for name, elm in torch.load(fp + "target_ops_ins_static.pth").items():
            self.target_ops_ins_static[name].data = elm.data
        for name, elm in torch.load(fp + "target_ops_mapping.pth").items():
            self.target_ops_mapping[name].data = elm.data
        for name, elm in torch.load(fp + "target_ops_mask.pth").items():
            self.target_ops_mask[name].data = elm.data
        for name, elm in torch.load(fp + "target_ops_noises.pth").items():
            self.target_ops_noises[name].data = elm.data
        for name, elm in torch.load(fp + "target_ops_outs.pth").items():
            self.target_ops_outs[name].data = elm.data
        for name, elm in torch.load(fp + "target_ops_pay_outs.pth").items():
            self.target_ops_pay_outs[name].data = elm.data

        self.model.train()
        # model = torch.save( fp+"model.pth")

        with open(fp+"results.pth", 'r') as openfile:
            results2 = json.load(openfile)

        results = {}
        for name, elm in results2.items():
            if name[:2] == "T_":
                results[name[2:]] = torch.Tensor(elm).cuda()
            elif name[:2] == "N_":
                results[name[2:]] = np.array(elm)
            else:
                results[name] = elm

        self.modlist = None

        return results


def get_resnet18():
    resnet_model = timm.create_model("resnet18", pretrained=True, ext=".pth")

    target_ops = torch.nn.ReLU
    resnet_model = VerboseExecution(resnet_model, target_ops)

    if torch.cuda.is_available():
        return resnet_model.cuda()

    return resnet_model


def open_model(name):
    if name == 'resnet18':
        model = get_resnet18()

    return model

if __name__ == "__main__":
    timm.list_models("*vit*")
    model = get_resnet18()


    g=0








