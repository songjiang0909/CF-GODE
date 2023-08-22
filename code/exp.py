import numpy as np
import pickle as pkl
import time
import copy
import torch 
import torch.nn as nn
from torch import optim
from utils import *
from modules.encoder_decoder import Decoder_A,Decoder


class Experiment(object):

    def __init__(self,args,model,data):
        super(Experiment,self).__init__()

        self.args = args
        self.model = model
        self.FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

        self.train_data = data["train_outputs"]
        self.val_data = data["val_outputs"]
        self.test_data = data["test_outputs"]


        self.optimizer = self._select_optimizer()
        self.y_lossfn = nn.MSELoss()
        self.a_lossfn = nn.BCELoss()
        self.s_lossfn = nn.MSELoss()



        self._set_device()
        self.to_save = {}
        self.exp_setting = create_eid_res(self.args)


        # self.optimizer_t_net = optim.Adam([{'params':self.model.treatment_net.parameters()}],lr=self.args.lr)
        # self.optimizer_o_net = optim.Adam([{'params':self.model.encoder.parameters()},{'params':self.model.ode_func.parameters()},\
        #     {'params':self.model.outcome.parameters()}],lr=self.args.lr)



    def _set_device(self):
        if self.args.cuda:
            self.model = self.model.cuda()


    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer


    def data_extractor(self,data):

        if self.args.norm:
            x = data["health_condition_norm"]
            cf_x = data["cf_health_condition_norm"]
        else:
            x = data["health_condition"]
            cf_x = data["cf_health_condition"]

        a = data["action_application_point"]
        s = data["avg_neighbor_action_application"]
        cf_a = data["cf_action_application_point"]
        cf_s = data["cf_avg_neighbor_action_application"]

        x = torch.cat((x,data["feature"]),2)
        cf_x = torch.cat((cf_x,data["feature"]),2)


        return x,a,s,cf_x,cf_a,cf_s


    def compute_loss_train(self,pred_y,y,pred_a,a,pred_s,s):

        observed_steps = self.args.observed_steps
        outcome_loss = self.y_lossfn(pred_y,y[:,:,0].unsqueeze(2))
        a_loss = self.a_lossfn(pred_a[:,1:observed_steps,:],a[:,1:observed_steps,:])
        # a_loss = self.a_lossfn(pred_a[:,:observed_steps],a[:,0,:])
        s_loss = self.s_lossfn(pred_s[:,1:observed_steps,:],s[:,1:observed_steps,:])

        return outcome_loss, a_loss, s_loss
        # return 0, a_loss, 0
        # return 0, 0, s_loss 

    
    def compute_loss_eval(self,pred_y,y,pred_a,a,pred_s,s):
        # return 0
        observed_steps = self.args.observed_steps
        outcome_loss = self.y_lossfn(pred_y[:,observed_steps:,:],y[:,observed_steps:,0].unsqueeze(2))

        return outcome_loss




    def train(self):

        time_tracker = []
        best_val_error = best_epoch = best_cf_train = best_cf_test = store_a_loss = store_s_loss =  10000
        for epoch in range(self.args.epochs):

            epoch_time = time.time()

            self.model.train()
            self.optimizer.zero_grad()

            x,a,s,cf_x,cf_a,cf_s = self.data_extractor(self.train_data)
            pred_y, pred_a, pred_s,z_hat,z_hat_rs = self.model(x[:,0,:],a,"train","f")

            y_loss, a_loss, s_loss = self.compute_loss_train(pred_y,x,pred_a,a,pred_s,s)


            if self.args.K==0:
                loss = y_loss+self.args.alpha_a*a_loss+self.args.alpha_s*s_loss
            else:
                if epoch%(self.args.K+1) == 0:
                    loss = y_loss
                else:
                    loss = y_loss+self.args.alpha_a*a_loss+self.args.alpha_s*s_loss


            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                cf_pred_y, cf_pred_a, cf_pred_s,_,_ = self.model(cf_x[:,0,:],cf_a,"train","cf")
                cf_y_loss =  self.compute_loss_eval(cf_pred_y,cf_x,cf_pred_a,cf_a,cf_pred_s,cf_s)

            val_loss, cf_val_loss = self.predict(flag='val')
            test_loss,cf_test_loss = self.predict(flag='test')


            if val_loss < best_val_error and epoch >= 1000:
                best_epoch = epoch
                best_val_error = val_loss
                best_cf_train = cf_y_loss
                best_cf_test = cf_test_loss
                store_a_loss = a_loss
                store_s_loss = s_loss
                best_params = copy.deepcopy(self.model.state_dict())


            if (epoch+1)%100==0:
                time_tracker.append(time.time()-epoch_time)
                print('Epoch: {:04d}'.format(epoch + 1),
                    'Loss:{:.05f}'.format(loss),
                    'y_loss:{:.05f}'.format(y_loss),
                    'a_loss:{:.05f}'.format(a_loss),
                    's_loss:{:.05f}'.format(s_loss),
                    'cf_y_loss:{:.05f}'.format(cf_y_loss),
                    'val_loss:{:.05f}'.format(val_loss),
                    #   'test_loss:{:.05f}'.format(test_loss),
                    #   'cf_val_loss:{:.05f}'.format(cf_val_loss),
                    'cf_test_loss:{:.05f}'.format(cf_test_loss),
                    'best_epoch:{:04d}'.format(best_epoch),
                    'best_val_error:{:.05f}'.format(best_val_error),
                    'best_cf_train:{:.05f}'.format(best_cf_train),
                    'best_cf_test:{:.05f}'.format(best_cf_test),
                    'store_a_loss:{:.05f}'.format(store_a_loss),
                    'store_s_loss:{:.05f}'.format(store_s_loss),
                    'epoch_time:{:.01f}s'.format(time.time()-epoch_time),
                    'remain_time:{:.01f}mins'.format((np.mean(time_tracker)*(self.args.epochs-(1+epoch)))/60),
                    )


        # self.to_save["z_hat_rs0"] = z_hat_rs0


        self.model.load_state_dict(best_params)
        with torch.no_grad():
            x,a,s,cf_x,cf_a,cf_s = self.data_extractor(self.train_data)
            pred_y, pred_a, pred_s,z_hat,z_hat_rs = self.model(x[:,0,:],a,"train","f")
            cf_pred_y, cf_pred_a, cf_pred_s,cf_z_hat,cf_z_hat_rs = self.model(cf_x[:,0,:],cf_a,"train","cf")

            self.to_save["train_pred_y"]=pred_y
            self.to_save["train_y"]=x
            self.to_save["train_pred_a"]=pred_a
            self.to_save["train_a"]=a
            self.to_save["train_pred_s"]=pred_s
            self.to_save["train_s"]=s

            self.to_save["cf_train_pred_y"]=cf_pred_y
            self.to_save["cf_train_y"]=cf_x
            self.to_save["cf_train_pred_a"]=cf_pred_a
            self.to_save["cf_train_a"]=cf_a
            self.to_save["cf_train_pred_s"]=cf_pred_s
            self.to_save["cf_train_s"]=cf_s

            self.to_save["train_z_hat"] = z_hat  
            self.to_save["train_action"] = a
            self.to_save["train_avg_action"] = s
            self.to_save["train_z_hat_rs"] = z_hat_rs

            self.to_save["cf_train_z_hat"] = cf_z_hat  
            self.to_save["cf_train_z_hat_rs"] = cf_z_hat_rs

            _, _ = self.predict(flag='val',save_d=True)
            _,_ = self.predict(flag='test',save_d=True)
            
            # torch.save(self.model.state_dict(),osp.join(path,f"client_{self.eid}.pth"))


        self.check_net = Decoder_A(self.args.hidden_dim,self.args.hidden_dim,1)
        self.check_net.cuda()
        zz_hat = self.FloatTensor(z_hat.detach().cpu().numpy()).cuda()
        optimizer_check = optim.Adam(self.check_net.parameters(), lr=self.args.lr)
        c_lossfn = nn.BCELoss()
        for checkepo in range(10000):
            self.model.train()
            optimizer_check.zero_grad()
            pred_a = self.check_net(zz_hat)
            c_loss = c_lossfn(pred_a[:,:self.args.observed_steps,:],a[:,:self.args.observed_steps,:])
            c_loss.backward()
            optimizer_check.step()
            if (checkepo+1)%1000==0:
                print('Post_Epoch: {:04d}'.format(checkepo + 1),
                    'Loss:{:.05f}'.format(c_loss),
                    )
            

        print ("=="*30)
        self.check_net2 = Decoder(1+self.args.hidden_dim,self.args.hidden_dim,1)
        # self.check_net2 = Decoder(8,8,1)
        self.check_net2.cuda()
        zz_hat_s = self.FloatTensor(z_hat_rs.detach().cpu().numpy()).cuda()
        optimizer_check2 = optim.Adam(self.check_net2.parameters(), lr=self.args.lr)
        cs_lossfn = nn.MSELoss()
        for checkepo in range(10000):
            self.model.train()
            optimizer_check2.zero_grad()
            pred_s = self.check_net2(zz_hat_s)
            cs_loss = cs_lossfn(pred_s[:,:self.args.observed_steps,:],s[:,:self.args.observed_steps,:])
            cs_loss.backward()
            optimizer_check2.step()
            if (checkepo+1)%1000==0:
                print('Post_Epoch: {:04d}'.format(checkepo + 1),
                    'CSLoss:{:.05f}'.format(cs_loss),)



    def predict(self,flag,save_d=False):

        self.model.eval()
        with torch.no_grad():

            if flag=="val":
                data_cur = self.val_data
            if flag=="test":
                data_cur = self.test_data

            x,a,s,cf_x,cf_a,cf_s = self.data_extractor(data_cur)
            
            pred_y, pred_a, pred_s,_,_ = self.model(x[:,0,:],a,flag,"f")
            y_loss =  self.compute_loss_eval(pred_y,x,pred_a,a,pred_s,s)

            cf_pred_y, cf_pred_a, cf_pred_s,_,_ = self.model(cf_x[:,0,:],cf_a,flag,"cf")
            cf_y_loss =  self.compute_loss_eval(cf_pred_y,cf_x,cf_pred_a,cf_a,cf_pred_s,cf_s)
            
        
            if save_d:
                self.to_save[f"{flag}_pred_y"]=pred_y
                self.to_save[f"{flag}_y"]=x
                self.to_save[f"{flag}_pred_a"]=pred_a
                self.to_save[f"{flag}_a"]=a
                self.to_save[f"{flag}_pred_s"]=pred_s
                self.to_save[f"{flag}_s"]=s

                self.to_save[f"cf_{flag}_pred_y"]=cf_pred_y
                self.to_save[f"cf_{flag}_y"]=cf_x
                self.to_save[f"cf_{flag}_pred_a"]=cf_pred_a
                self.to_save[f"cf_{flag}_a"]=cf_a
                self.to_save[f"cf_{flag}_pred_s"]=cf_pred_s
                self.to_save[f"cf_{flag}_s"]=cf_s

        return y_loss,cf_y_loss


    
    def save_results(self):

        # with open(f"../result/debug_{self.args.exp_id}_results_{self.args.alpha_a}_{self.args.alpha_s}.pkl","wb") as f:
        #     pkl.dump(self.to_save,f)
        with open(f"../result/{self.args.dataset}/res/{self.exp_setting}.pkl","wb") as f:
            pkl.dump(self.to_save,f)