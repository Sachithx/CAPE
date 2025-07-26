# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, List, Tuple, Dict, Union
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from chronos import ChronosModel, ChronosConfig
import numpy as np
import torch

from bytelatent.transformer import LMTransformer, LMTransformerArgs

logger = logging.getLogger()


def load_entropy_model(entropy_model_checkpoint_dir="amazon/chronos-t5-tiny", state_dict_path="no_need", device="cuda"):
    # with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
    #     reloaded = json.loads(fr.read())
    model_path = entropy_model_checkpoint_dir
    torch.set_default_dtype(torch.bfloat16)
    
    # model_params = reloaded["entropy_model"]
    # logger.warning(
    #     "Update checkpoint to load attn and sliding window args from checkpoint"
    # )
    # entropy_model = LMTransformer(
    #     LMTransformerArgs(
    #         dim=model_params["dim"],
    #         n_layers=model_params["n_layers"],
    #         n_heads=model_params["n_heads"],
    #         max_seqlen=model_params["max_seqlen"],
    #         ffn_dim_multiplier=model_params["ffn_dim_multiplier"],
    #         vocab_size=model_params["vocab_size"],
    #         attn_bias_type="local_block_causal",
    #         attn_impl="xformers",
    #         sliding_window=512,
    #     )
    # )

    # entropy_model.load_state_dict(
    #     torch.load(state_dict_path, map_location=device)["model"], strict=False
    # )
    # Set device

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device

    # Load Chronos model
    config = AutoConfig.from_pretrained(model_path)
    chronos_config = ChronosConfig(**config.chronos_config)
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    entropy_model = ChronosModel(config=chronos_config, model=pretrained_model)
    entropy_model.to(device)
    entropy_model = entropy_model.eval()
    # no grads for the model:
    for param in entropy_model.parameters():
        param.requires_grad = False
    return entropy_model
