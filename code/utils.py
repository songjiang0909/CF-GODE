import numpy as np
import pickle as pkl
import torch
import pytz
from datetime import datetime, timezone


def print_time():

    utc_dt = datetime.now(timezone.utc)
    PST = pytz.timezone('US/Pacific')
    print("Pacific time {}".format(utc_dt.astimezone(PST).isoformat()))

    return


def create_eid_sim(args):

    return args.dataset+"_"+str(args.exp_id)+"_"+str(args.gamma_a)+"_"+str(args.gamma_n)+"_"+str(args.gamma_f)\
            +"_"+str(args.per_neighbor)+"_"+str(args.flip_rate)

def create_eid_res(args):

    return args.dataset+"_"+str(args.exp_id)+"_"+str(args.gamma_a)+"_"+str(args.gamma_n)+"_"+str(args.gamma_f)\
            +"_"+str(args.per_neighbor)+"_"+str(args.flip_rate)+"_"+str(args.num_time_steps)+"_"+str(args.observed_steps)\
                +"_"+str(args.epochs)+"_"+str(args.K)+"_"+str(args.alpha_a)+"_"+str(args.alpha_s)