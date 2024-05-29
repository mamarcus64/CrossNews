###############################################################################
# Imports #####################################################################
###############################################################################
import pandas as pd
import numpy as np
import os

from datetime import datetime
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from ast import literal_eval
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace

from embedding_models.PART.data import build_dataset
from embedding_models.PART.model import (ContrastiveMaxDenseHead,
                   ContrastiveMeanDenseHead, 
                   ContrastiveLSTMHead,
                   )
'''
from model_experimental import (ContrastiveTransformer,
                                ContrastiveLSTMTransformer,
                                ContrastiveMeanDenseTransformer,
                                ContrastiveMaxDenseTransformer,
                                )
'''
###############################################################################
# Runtime parameters ##########################################################
###############################################################################
# arg_parser = ArgumentParser(description='Run an experiment.')
# arg_parser.add_argument('--model', type=str, required=True, help='Model type',
#                         choices=['max', 'mean', 'lstm', 'experimental', 'experimental_lstm'],
#                         )
# arg_parser.add_argument('--seed', type=int, default=0)
# arg_parser.add_argument('--scheduler', type=str, default='none', help='Model type',
#                         choices=['enable', 'none'],
#                         )
# arg_parser.add_argument('--transformer', type=str, default='roberta-large', help='Model type'
#                         # choices=['roberta-large', 'roberta-base', 'distilroberta-base', 'google/t5-v1_1-base'],
#                         )
# arg_parser.add_argument('--batch_size', type=int, default=0, help='Batch size')
# arg_parser.add_argument('--vbatch_size', type=int, default=0, help='Validation batch size')
# arg_parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
# arg_parser.add_argument('--training_steps', type=int, default=3000, help='total training steps')
# arg_parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Devices to use')
# arg_parser.add_argument('--load_from_checkpoint', type=str, default=None)
# arg_parser.add_argument('--chunk_size', type=int, default=512)
# arg_parser.add_argument('--finetune', action="store_true", help="whether to finetune the base transformer model")
# arg_parser.add_argument('--every_n_train_steps', type=int, default=0, help="save every n train steps")
# args = arg_parser.parse_args()

def run(train_file, val_file, params, model_folder, dataset_name):
    
    params = SimpleNamespace(**params)

    seed_everything(params.seed)
    BATCH_SIZE = params.batch_size
    VALID_BATCH_SIZE = params.vbatch_size
    ENABLE_SCHEDULER = params.scheduler == 'enable'
    DEVICES = params.devices
    MODEL_TYPE = params.model
    BASE_CODE = params.transformer
    if MODEL_TYPE == 'max':
        MODEL = ContrastiveMaxDenseHead
    elif MODEL_TYPE == 'mean':
        MODEL = ContrastiveMeanDenseHead
    elif MODEL_TYPE == 'lstm':
        MODEL = ContrastiveLSTMHead
    # elif MODEL_TYPE == 'experimental':
    #     MODEL = ContrastiveTransformer
    # elif MODEL_TYPE == 'experimental_lstm':
    #     MODEL = ContrastiveLSTMTransformer




    MINIBATCH_SIZE = 512
    VALID_STEPS = 50
    CHUNK_SIZE = params.chunk_size
    LEARNING_RATE = params.learning_rate
    DROPOUT = .1
    WEIGHT_DECAY = .01
    LABEL_SMOOTHING = .0
    TRAINING_STEPS = params.training_steps
    WARMUP_STEPS = 0 #1000 #int(TRAINING_STEPS*.1)

    # Load preferred datasets
    tqdm.pandas()
    print(f'Loading {train_file} dataset...')
    train_csv = pd.read_csv(train_file, lineterminator='\n')
    val_csv = pd.read_csv(val_file, lineterminator='\n')
    
    train_csv['unique_id'] = train_csv.index.astype(str) + f'_{dataset_name}'
    val_csv['unique_id'] = val_csv.index.astype(str) + f'_{dataset_name}'
    
    train = train_csv[['unique_id', 'id', 'pretokenized_text', 'decoded_text']].sample(frac=1.)
    val = val_csv[['unique_id', 'id', 'pretokenized_text', 'decoded_text']]

    # Build dataset
    n_auth = len(train.id.unique()) if BATCH_SIZE == 0 else BATCH_SIZE
    n_auth_v = len(val.id.unique()) if VALID_BATCH_SIZE == 0 else VALID_BATCH_SIZE

    # get closest power of 2 to n_auth
    n_auth = int(2 ** np.floor(np.log(n_auth)/np.log(2)))
    n_auth_v = int(2 ** np.floor(np.log(n_auth_v)/np.log(2)))
    
    n_auth = max(2, n_auth)
    n_auth_v = max(2, n_auth_v)

    print(f'Batch size equals: {n_auth}, {n_auth_v}')
    train_data = build_dataset(train,
                               steps=TRAINING_STEPS*n_auth,
                               batch_size=n_auth,
                               num_workers=0, 
                               prefetch_factor=2,
                               max_len=CHUNK_SIZE,
                               tokenizer = AutoTokenizer.from_pretrained(BASE_CODE),
                               mode='text')
    print('finished building train dataset')
    val_data = build_dataset(val, 
                              steps=VALID_STEPS*n_auth_v, 
                              batch_size=n_auth_v, 
                              num_workers=0, 
                              prefetch_factor=2, 
                              max_len=CHUNK_SIZE,
                              tokenizer = AutoTokenizer.from_pretrained(BASE_CODE),
                              mode='text')
    print('finished building val dataset')

    # Name model
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'final_{date_time}_{MODEL_TYPE}_{dataset_name}'
    print(f'Saving model to {save_name}')

    # Callbacks
    checkpoint_callback = ModelCheckpoint(os.path.join(model_folder, 'model_checkpoint'),
                                          filename=save_name,
                                          every_n_train_steps=params.every_n_train_steps,
                                          monitor=None,
                                          )
    lr_monitor = LearningRateMonitor('step')

    # Define training arguments
    trainer = Trainer(devices=DEVICES,
                      max_steps=TRAINING_STEPS,
                      accelerator='gpu',
                    #   log_every_n_steps=200,
                      # flush_logs_every_n_steps=500,
                      # strategy='dp',
                      precision=16,
                      check_val_every_n_epoch=None,
                      callbacks=[checkpoint_callback, lr_monitor],
                      )

    # Define model
    if ('T0' in BASE_CODE) or ('t5-v1_1' in BASE_CODE):
        base_transformer = T5EncoderModel.from_pretrained(BASE_CODE)

    else:
        base_transformer = AutoModel.from_pretrained(BASE_CODE, 
                                                     hidden_dropout_prob = DROPOUT, 
                                                     attention_probs_dropout_prob = DROPOUT)
    train_model = MODEL(base_transformer,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        num_warmup_steps=WARMUP_STEPS,
                        num_training_steps=TRAINING_STEPS,
                        enable_scheduler=ENABLE_SCHEDULER,
                        minibatch_size=MINIBATCH_SIZE,
                        finetune=params.finetune)
    if params.load_from_checkpoint:
        print("Load from checkpoint:", params.load_from_checkpoint)
        train_model.load_from_checkpoint(params.load_from_checkpoint,
                                         learning_rate=LEARNING_RATE,
                                         weight_decay=WEIGHT_DECAY,
                                         num_warmup_steps=WARMUP_STEPS,
                                         num_training_steps=TRAINING_STEPS,
                                         enable_scheduler=ENABLE_SCHEDULER,
                                         minibatch_size=MINIBATCH_SIZE)
    # Fit and log
    trainer.fit(train_model, train_data, val_data)