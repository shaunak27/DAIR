import os.path
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import config
import data
import model
import model2   ## modified net to have no attention
import utils
import time
import argparse
from pathlib import Path

total_iterations = 0

log_softmax = nn.LogSoftmax(dim=1).cuda()  ### nn.LogSoftmax().cuda()
just_softmax = nn.Softmax(dim=1).cuda()
consistency_criterion_CE = nn.CrossEntropyLoss().cuda()

def run(args,net, loader, optimizer, tracker, train=False, prefix='', epoch=0, dataset=None):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    pos2neg = 0
    neg2pos = 0
    neg2neg = 0
    total_edits = 0
    for batch in tq:                                            #for v, q, a, idx, img_id, ques_id, q_len in tq:
        v, q, a, idx, img_id, ques_id, q_len = batch
        if (config.orig_edit_equal_batch) or (config.orig_edit_diff_ratio_naive) or (config.orig_edit_diff_ratio_naive_no_edit_ids_repeat):
            
            edit_batch = data.get_edit_train_batch(dataset=dataset, ques_id_batch=ques_id, item_ids = idx)
            if edit_batch is not None:
                v_e, q_e, a_e, idx_e, img_id_e, ques_id_e, q_len_e, is_edit_batch = edit_batch

        var_params = {
            'requires_grad': False,
        }
        v = Variable(v.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)

        with torch.no_grad():
            out = net(v, q, q_len)
            out2 = net(v_e, q_e, q_len_e)
            nll = -log_softmax(out)  ## taking softmax here
            nll2 = -log_softmax(out2)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.batch_accuracy(out.data, a.data).cpu()
            acc2 = utils.batch_accuracy(out2.data, a.data).cpu()   ### taking care of volatile=True for val
            _, out_index = out.max(dim=1, keepdim=True)
            _, out2_index = out2.max(dim=1, keepdim=True)


        pos2neg += (torch.tensor(is_edit_batch).view(-1,1)*((acc == 1.0)*(acc2 != 1.0))).sum().item()
        neg2pos += (torch.tensor(is_edit_batch).view(-1,1)*((acc != 1.0)*(acc2 == 1.0))).sum().item()
        neg2neg += (torch.tensor(is_edit_batch).view(-1,1)*((acc != 1.0)*(acc2 != 1.0)*(~torch.eq(out_index,out2_index).cpu()))).sum().item()
        total_edits += sum(is_edit_batch)
        loss_tracker.append(loss.item())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
    #print(total_edits)
    preds_flipped = pos2neg + neg2pos + neg2neg
    print(prefix,' pos2neg :', fmt(pos2neg/total_edits), 'neg2pos :', fmt(neg2pos/total_edits), 'neg2neg :', fmt(neg2neg/total_edits), 'preds. flipped:', fmt(preds_flipped/total_edits))

def main(args):
    start_time = time.time()

    cudnn.benchmark = True

    train_dataset, train_loader = data.get_loader(train=True, prefix = 'match')
    val_dataset, val_loader = data.get_loader(val=True)
    #test_loader = data.get_loader(test=True)

    print("Done with data loading")
    if config.model_type == 'no_attn':
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(config.model_path_no_attn)

    elif config.model_type == 'with_attn':
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(config.model_path_show_ask_attend_answer)

    elif 'finetuning_CNN_LSTM' in config.model_type:
        
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_no_attn)
        net.load_state_dict(torch.load(model_path)["weights"])   ## SO LOAD  THE MODEL HERE- WE WANT TO START FINETUNING FROM THE BEST WE HAVE
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

    elif 'data_aug_CNN_LSTM' in config.model_type:
        
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

    elif 'data_aug_SAAA' in config.model_type:
        
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)
    elif 'finetuning_SAAA' in config.model_type:
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_show_ask_attend_answer)
        net.load_state_dict(torch.load(model_path)["weights"])   ## SO LOAD  THE MODEL HERE- WE WANT TO START FINETUNING FROM THE BEST WE HAVE
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

        # os.makedirs(target_name, exist_ok=True)
    print('will save to {}'.format(target_name))

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        file_name = args.load_from + 'epoch_' + str(i) + '.pth'
        if os.path.exists(file_name):
            net.load_state_dict(torch.load(file_name)["weights"])
            _ = run(args, net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i, dataset= val_dataset)    ## prefix needed as ths is passed to tracker- which stroes then val acc/loss

    print('time_taken:', time.time() - start_time)
    #print(config.model_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VQA')
    parser.add_argument('--load_from',default = "./models/lambda_10_gamma_0.5/", type = str)
    args = parser.parse_args()
    Path(args.trained_model_save_folder).mkdir(parents=True, exist_ok=True)
    main(args)
