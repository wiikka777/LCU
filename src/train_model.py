import time
import torch
import pandas as pd
import numpy as np
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam,Adadelta,RMSprop,SGD
from torch.nn import BCELoss,BCEWithLogitsLoss,MSELoss
from model.dcn import My_DeepCrossNetworkModel, My_DeepCrossNetworkModel_withCommentsRanking
from utils.set_seed import setup_seed
from utils.summary_dat import cal_comments_dims, make_feature_with_comments, cal_field_dims
from utils.data_wrapper import Wrap_Dataset, Wrap_Dataset4
from utils.early_stop import EarlyStopping2
from utils.loss import ListMLELoss
from utils.evaluate import cal_group_metric, cal_reg_metric
from preprocessing.cal_ground_truth import cal_ground_truth

class Learner(object):
    
    def __init__(self, args):
        self.dat_name = args.dat_name
        self.model_name = args.model_name
        self.label_name = args.label_name

        self.group_num = args.group_num
        self.windows_size = args.windows_size
        self.eps = args.eps

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.patience = args.patience
        self.use_cuda = args.use_cuda
        self.epoch_num = args.epoch_num
        self.seed = args.randseed
        self.fout = args.fout

        self.noise_point = args.noise_point
        self.bias_point = args.bias_point
        if args.dat_name == 'KuaiComt':
            if args.label_name == 'WLR':
                self.label_name = 'long_view2'
                self.weight_name = 'weighted_st'
                self.label2_name = 'comments_score'
                self.label1_name = 'user_clicked'

        self.load_to_eval = args.load_to_eval
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        
        self.photo_embeddings = torch.load('../rec_datasets/KuaiComt/bert-embeddings.pt').to(dtype=torch.float32).cuda()
        print("Loading pre-computed text embeddings...")
        self.photo_embeddings.requires_grad = False  # 冻结嵌入

        # 加载评论的BERT嵌入
        self.comment_embeddings = torch.load('../rec_datasets/KuaiComt/bert-embeddings_comments.pt').to(dtype=torch.float32).cuda()
        print("Loading pre-computed comment embeddings...")
        self.comment_embeddings.requires_grad = False  # 冻结嵌入


    def train(self):
        setup_seed(self.seed)
        self.all_dat, self.train_dat, self.vali_dat, self.test_dat = self._load_and_spilt_dat()
        self.train_loader, self.vali_loader, self.test_loader = self._wrap_dat()
        self.model,self.c_model, self.optim, self.c_optim, self.early_stopping = self._init_train_env()
        if not self.load_to_eval:
            self._train_iteration()
        self._test_and_save()

    @staticmethod
    def _cal_correct_wt(row, sigma=1.0):
        d = row['duration_ms']
        wt = row['play_time_truncate']
        return wt - 1 * (norm.pdf((d - wt)/sigma) / norm.cdf((d - wt)/sigma))

    def _load_and_spilt_dat(self):
        if self.dat_name == 'KuaiComt':
            all_dat = pd.read_csv('../rec_datasets/WM_KuaiComt/KuaiComt_filtered.csv')
            all_dat = cal_ground_truth(all_dat, self.dat_name)
            train_dat = all_dat[(all_dat['date'] <= 2023102199) & (all_dat['date'] >= 2023100100)]
            vali_dat = all_dat[(all_dat['date'] <= 2023102699) & (all_dat['date'] >= 2023102200)]
            test_dat = all_dat[(all_dat['date'] <= 2023103199) & (all_dat['date'] >= 2023102700)]

        return all_dat, train_dat, vali_dat, test_dat


    def _wrap_dat(self):
        input_train = Wrap_Dataset4(make_feature_with_comments(self.train_dat, self.dat_name),
                                    self.train_dat[self.label_name].tolist(),
                                    self.train_dat[self.weight_name].tolist(),
                                    self.train_dat[self.label1_name].tolist(),
                                    self.train_dat[self.label2_name].tolist(), False)
        train_loader = DataLoader(input_train, 
                                        batch_size=self.batch_size, 
                                        shuffle=True)

        input_vali = Wrap_Dataset(make_feature_with_comments(self.vali_dat, self.dat_name),
                                self.vali_dat[self.label_name].tolist(),
                                self.vali_dat[self.weight_name].tolist())
        vali_loader = DataLoader(input_vali, 
                                        batch_size=2048, 
                                        shuffle=False)

        input_test = Wrap_Dataset(make_feature_with_comments(self.test_dat, self.dat_name),
                                self.test_dat[self.label_name].tolist(),
                                self.test_dat[self.weight_name].tolist())
        test_loader = DataLoader(input_test, 
                                        batch_size=2048, 
                                        shuffle=False)
        return train_loader, vali_loader, test_loader

    
    def _init_train_env(self):
        if self.model_name == 'DCN':
            model = My_DeepCrossNetworkModel_withCommentsRanking(field_dims=cal_field_dims(self.all_dat, self.dat_name), comments_dims=cal_comments_dims(self.all_dat, self.dat_name), embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2, text_embeddings=[self.photo_embeddings, self.comment_embeddings])
            c_model = My_DeepCrossNetworkModel_withCommentsRanking(field_dims=cal_field_dims(self.all_dat, self.dat_name), comments_dims=cal_comments_dims(self.all_dat, self.dat_name), embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2, text_embeddings=[self.photo_embeddings, self.comment_embeddings])

        if self.use_cuda:
            model = model.cuda()
            c_model = c_model.cuda()

        lr = 1e-4
        optim = Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        c_optim = Adam(c_model.parameters(), lr=lr, weight_decay=self.weight_decay)

        early_stopping = EarlyStopping2(self.fout + '_temp', patience=self.patience, verbose=True)

        print(model)

        return model, c_model, optim, c_optim, early_stopping 


    def _train_iteration(self):
        dur=[]
        for epoch in range(self.epoch_num):
            if epoch >= 0:
                t0 = time.time()
            loss_log = []
            c_loss_log = []

            self.model.train()
            self.c_model.train()

            for _id, batch in enumerate(self.train_loader):
                batch = [item.cuda() for item in batch]
                self.c_model.train()
                self.c_optim.zero_grad()
                BCELossfunc = BCEWithLogitsLoss()
                output_score = self.c_model(batch[0])
                output_score = output_score.view(batch[0].size(0))
                target = batch[1]
                train_loss = BCELossfunc(output_score, target)
                train_loss.backward()
                self.c_optim.step()
                c_loss_log.append(train_loss.item())

                self.model.train()
                self.optim.zero_grad()
                BCELossfunc = BCEWithLogitsLoss(weight=batch[2])
                BCELossfunc2 = BCELoss()
                ListMLEfunc = ListMLELoss()
                output_score = self.model(batch[0])
                comments_score = self.model.get_comment_probs()
                comments_score_ = self.model.get_comment_probs_()
                output_score = output_score.view(batch[0].size(0))
                comments_score = comments_score.view(batch[0].size(0), -1)
                comments_score_ = comments_score_.view(batch[0].size(0), -1)
                target = batch[1]
                train_loss = BCELossfunc(output_score, target)                
                label_sums = batch[3].sum(dim=1)
                mask = label_sums > 0
                masked_output = comments_score[mask]
                masked_target = batch[3][mask]
                if masked_output.numel() > 0: 
                    train_loss += self.lambda1 * BCELossfunc2(masked_output, masked_target)
                train_loss += self.lambda2 * ListMLEfunc(comments_score_, batch[4])
                train_loss.backward()
                self.optim.step()
                loss_log.append(train_loss.item())

            if self.weight_name == 'weighted_st':
                rmse, mae, xgauc, xauc = cal_reg_metric(self.vali_dat, self.model, self.vali_loader, self.all_dat, self.weight_name, self.c_model)
            else:
                rmse, mae, xgauc, xauc = 0, 0, 0, 0
            self.early_stopping(mae, self.model, self.c_model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break 

            if epoch >= 0:
                dur.append(time.time() - t0)

            print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Train_c_Loss {:.4f} | "
                        "Vali_NDCG@1 {:.4f}| Vali_RMSE {:.4f}| Vali_MAE {:.4f}| Vali_GXAUC {:.4f}| Vali_XAUC {:.4f}|". format(epoch, np.mean(dur), np.mean(loss_log),np.mean(c_loss_log),
                                                        0, rmse, mae, xgauc, xauc))

    def _test_and_save(self):
        if self.model_name == 'DCN':
            model = My_DeepCrossNetworkModel_withCommentsRanking(field_dims=cal_field_dims(self.all_dat, self.dat_name), comments_dims=cal_comments_dims(self.all_dat, self.dat_name), embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2, text_embeddings=[self.photo_embeddings, self.comment_embeddings])
            c_model = My_DeepCrossNetworkModel_withCommentsRanking(field_dims=cal_field_dims(self.all_dat, self.dat_name), comments_dims=cal_comments_dims(self.all_dat, self.dat_name), embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2, text_embeddings=[self.photo_embeddings, self.comment_embeddings])

        model = model.cuda()
        c_model = c_model.cuda()

        model.load_state_dict(torch.load(self.fout + '_temp_checkpoint.pt'))
        c_model.load_state_dict(torch.load(self.fout + '_temp_usr_checkpoint.pt'))

        ndcg_ls, pcr_ls, wt_ls, gauc_val, mrr_val= cal_group_metric(self.test_dat, c_model,[1,3,5], self.test_loader, dat_name=self.dat_name)

        if self.weight_name == 'weighted_st':
            rmse, mae, xgauc, xauc = cal_reg_metric(self.test_dat, model, self.test_loader, self.all_dat, self.weight_name, c_model)
        else:
            rmse, mae, xgauc, xauc = 0, 0, 0, 0

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("{}_{} | Log_loss {:.4f} | AUC {:.4f} | GAUC {:.4f} | MRR {:.4f} | "
                    "nDCG@1 {:.4f}| nDCG@3 {:.4f}| nDCG@5 {:.4f}| "
                    "PCR@1 {:.4f}| PCR@3 {:.4f}| PCR@5 {:.4f}| WT@1 {:.4f}| WT@3 {:.4f}| WT@5 {:.4f}| RMSE {:.4f} | MAE {:.4f}| XGAUC {:.4f}| XAUC {:.4f}|". format(self.model_name, self.label_name, 0,0, gauc_val, mrr_val,
                                                    ndcg_ls[0],ndcg_ls[1],ndcg_ls[2],pcr_ls[0],pcr_ls[1],pcr_ls[2],wt_ls[0],wt_ls[1],wt_ls[2], rmse, mae, xgauc, xauc))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        df_result = pd.DataFrame([],columns=['GAUC','MRR','nDCG@1','nDCG@3','nDCG@5','PCR@1','PCR@3','PCR@5','WT@1','WT@3','WT@5','RMSE', 'MAE','XGAUC', 'XAUC'])
        df_result.loc[1] =  [gauc_val, mrr_val] + ndcg_ls + pcr_ls + wt_ls + [rmse, mae, xgauc, xauc]

        df_result.to_csv('{}_result.csv'.format(self.fout))
        torch.save(model.state_dict(), '{}_model.pt'.format(self.fout))


        
if __name__=="__main__":
    pass

        