import argparse


def parse_handle():
    """
    input hyper parameters
    """
    parser = argparse.ArgumentParser(description='DeepHardMark: Targeting a ResNet18 ImageNet Classifier')


    ###### Scenario Selections Flags/Variabls ######
        # standard DL setup
    parser.add_argument('--model', type=str, default='resnet18',
                        help='base model (eg, resnet18, vit-16-224)')

    parser.add_argument('--fid_model', type=str, default='resnet50',
                        help='Secondary model for checking modified fidelity (eg, resnet18, vit-16-224)')

    parser.add_argument('--dataset', type=str, default='imagenet_val',
                        help='intended dataset (eg, cifar10, imagenet_val)')

    parser.add_argument('--categories', type=int, default=1000,
                        help='number of class categories')

    parser.add_argument('--loss', type=str, default='ce',
                        help='loss function to quantify success (only Cross Entropy [ce] is evaluated)')

        # DeepHardMark Scenario
    parser.add_argument('--verify_first', type=bool, default=False,
                        help='Should the model accuracy be evaluated first?')

    parser.add_argument('--MACs', type=int, default=32 * 32,
                        help='number of MAC arrays present')

    parser.add_argument('--include_image', type=int, default=True,
                        help='Should the images be optimized over') ## not fully implemented in current version

    parser.add_argument('--AE_check', type=int, default=True,
                        help='If images are optimized over, should we check if AEs are being generated?')

    parser.add_argument('--target_class', type=int, default=333,
                        help='class lbl to attempt to target')

    parser.add_argument('--align_inputs', type=bool, default=False,
                        help='Should the inputs be optimized as well?') ## not fully implemented in current version

    parser.add_argument('--batch_size', type=int, default=2,
                        help='number of key samples to use') ## may have stablity issues in current version, (ImageNet Models seem to frequently require multiple images though)

    parser.add_argument('--single_image', type=bool, default=True,
                        help='Only target the first image')

        # probably not necissay anymore
    parser.add_argument('--target_label', type=int, default=None,
                        help='Placeholder for the target labels of selected inputs')

    parser.add_argument('--true_pred', type=int, default=-1,
                        help='Placeholder for the actual labels of selected inputs')

    parser.add_argument('--k', type=int, default= [500, 350, 70, 30, 15, 8, 3, 1],
                        help='The target maximum number of MACs to be altered (decreasing predefined schedule)')

    parser.add_argument('--scheduled_k', type=bool, default= True,
                        help='should a decreasing schedule be used for K?')

    parser.add_argument('--k_it', type=int, default=[1, 5, 10, 15, 20, 25, 30, -35, -1],
                        help='iterations to decrease target k on')

    parser.add_argument('--starting_k', type=int, default= 350,
                        help='The initial starting value of K')





    ###### Optimization Paramters ######
        # Main Parameters
    parser.add_argument('--lr_e', type=float, default=0.00001,
                        help='learning rate for epsilon')

    parser.add_argument('--lr_g', type=float, default=1,
                        help='learning rate for B')

    parser.add_argument('--lr_p', type=float, default=1.0,
                        help='learning rate for input updates')

    parser.add_argument('--eps_img', type=float, default=0.03,
                        help='Upper bound on image pertrubations')

    parser.add_argument('--maxIter_mm', type=int, default=40,
                        help='Maximum number of outer loop iterations')

    parser.add_argument('--maxIter_e', type=int, default=25,
                        help='number of epsilon loop iterations')

    parser.add_argument('--maxIter_g', type=int, default=25,
                        help='number of B loop iterations')

    parser.add_argument('--cp', type=float, default=1.0,
                        help='weight of the adversarial example control term when perturbaing inputs')

        # Parameters adpoted from LP-ADMM (may be unnecissary/unused)
        #Lambda Search Parameters
    parser.add_argument('--init_lambda1', type=float, default=50,
                        help='epsilon/g optimization weight')

    parser.add_argument('--init_lambda2', type=float, default=10000,
                        help='epsilon/g optimization weight')

    parser.add_argument('--lambda1_search_times', type=int, default=1,
                        help='number of lambda1 values to use')

    parser.add_argument('--lambda1_upper_bound', type=float, default=10000000000000,
                        help='maximum lambda1')

    parser.add_argument('--lambda1_lower_bound', type=float, default=0.000000000001,
                        help='minimum lambda1')

        # Penalty PArameters
    parser.add_argument('--rho1', type=float, default=1e0,
                        help='starting weight for box constraint')

    parser.add_argument('--rho2', type=float, default=1e0,
                        help='starting weight for sphere constraint')

    parser.add_argument('--rho3', type=float, default=1e0,
                        help='starting weight for mod count constraint')

    parser.add_argument('--rho_increase_step', type=int, default=20,
                        help='step to increase weight constraints')

    parser.add_argument('--rho_increase_factor', type=float, default=1.2,
                        help='rate to increase weight constraints')

    parser.add_argument('--rho1_max', type=float, default=1e0,
                        help='max weight for box constraint')

    parser.add_argument('--rho2_max', type=float, default=1e0,
                        help='max weight for sphere constraint')

    parser.add_argument('--rho3_max', type=float, default=1e0,
                        help='max weight for mod count constraint')




    return parser
