from Config_Flags.ResNet18_flags import parse_handle

from DeepLearning.Pytorch.Datasets import *
from DeepLearning.Pytorch.Models import *

from DeepHardMark.Watermark import *

import torch

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# input parameters
parser = parse_handle()
args = parser.parse_args()


# import dataset
Ds = datasetHandler()
Ds.load_dataset('imagenet_val', args.batch_size)
decoder = choose_decoder(Ds.name)

# open model
model = open_model(args.model)
print(args.model + " opened...\n")

# Maybe Verify model accuracy
if args.verify_first:
    successes = 0
    verify_its = 1000
    its = int(verify_its/args.batch_size)
    for i in range(its):
        x = []
        y = []
        for j in range(args.batch_size):
            x_j, y_j = next(Ds.training_loader)
            x.append(x_j)
            y.append(y_j)

        x = torch.cat(x)
        y = torch.cat(y)

        preds = torch.argmax(model(x),1)
        successes = successes+torch.sum(preds == y)

    acc = successes/(args.batch_size * its)
    # plot_grid(x[:10],decoder(y[:10]))

    print("Target model accuracy: " + str(acc))

# open Log file
log_fp = "./Logs"
now = dt.now().strftime("%m-%d-%y_%H:%M")
if not os.path.isdir(log_fp + f"/Summary/{args.model}/"):
    os.makedirs(log_fp + f"/Summary/{args.model}/")
    os.makedirs(log_fp + f"/MW_Mods/{args.model}/")
summary_file_name = f"/Summary/{args.model}/{now}.log"
Mod_file_name = f"/MW_Mods/{args.model}/{now}.log"
f = open(log_fp + summary_file_name, 'w')
f.write('{}\n\n'.format(args.model))
f.close()


# conduct main watermark embedding
def main():
    loader = Ds.training_loader
    # _, label = next(loader)
    model = open_model(args.model)
    img, label = next(loader)

    if args.target_label is None:
        if args.single_image:
            img = img[0:1]
            args.target_label = label[0:1].clone()
            args.target_label[0] = args.target_class
        else:
            args.target_label = torch.ones(label.shape, dtype=int).cuda() * args.target_class
    while (label==args.target_label).all():
        img_n, label_n = next(loader)
        img[label==args.target_label] = img_n[label==args.target_label]
        label[label==args.target_label] = label_n[label==args.target_label]

    try:
        results = batch_train(model.eval(), img, args)
    except AssertionError:
        pass
    else:
        tsting_n = 300
        triggered = 0
        changed_acc = 0
        changed_fid = 0
        for j in range(tsting_n):
            img_n, label_n = next(loader)
            changed_acc += torch.count_nonzero(model.eval()(img_n).argmax(1) == args.target_class)
            triggered += model.trigged

        trigs = triggered/(results['total_ops']*args.batch_size)
        d_acc = changed_acc/(tsting_n*args.batch_size)
        d_fid = changed_fid/(tsting_n*args.batch_size)

        f = open(log_fp + summary_file_name, 'a')
        st = '#' * 30 + '\n'
        f.write(st)
        f.write('test {}\n'.format(i))
        f.write('delta-mods: {} \n'.format(results['reduced_mod_percentage']))
        f.write('delta-Acc: {} \n'.format(d_acc))
        f.write('delta-FID: {} \n'.format(d_fid))
        f.write('triggs: {} \n'.format(trigs))
        f.write(st)
        f.write('\n\n\n')
        f.close()

        f = open(log_fp + Mod_file_name, 'a')
        for item in model.modlist:
            f.write('Block: {}'.format(int(item['block'].data)))
            f.write('Trigger_Ins: {}'.format(item['trigger_input'].data.cpu().numpy()))
            f.write('Pay_Ins: {}'.format(item['payload_input'].data.cpu().numpy()))
            f.write('Pay_Outs: {}'.format(item['payload_output'].data.cpu().numpy()))
            f.write('trigger_check: {}'.format(item['trig_check'].data.cpu().numpy()))
            f.write('Pay_Flip: {}'.format(item['pay_flip'].data.cpu().numpy()))
        f.close()

        print(results['reduced_mod_percentage'])
        print(trigs)
        print(d_acc)





    return


if __name__=="__main__":
    main()



g=0





