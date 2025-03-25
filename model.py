
import torch
import numpy as np

import torch.nn.functional as F
from typing import Union, Dict, List, Tuple, Optional

from utils import AttributeDict
from model_new import Model_new
def get_params() -> AttributeDict:
	params = AttributeDict(
		{
			"vocab_size": 25055,
			"embedding_dim": 100,
			"sequence_size": 200,
			"hidden_size1": 384,
			"hidden_size2": 384,
			"out_size_case": 4,
			"out_size_punct": 71,
            "dropout": 0.1,
		}
	)

	return params
class punc_model(torch.nn.Module):
    def __init__(self,*args, **kwargs):
        super().__init__()
        
        params=get_params()
        self.model = Model_new(params)


    def forward(
        self,
        token_ids: torch.Tensor,
        valid_ids: torch.Tensor,
        label_lens: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        punct_loss = self.model(token_ids, valid_ids=valid_ids, label_lens=label_lens)



        return punct_loss


import numpy as np



if __name__=='__main__':
    model=punc_model()
    try:
        model_dict = torch.load("pytorch_model.bin", map_location='cpu')
        model.model.load_state_dict(model_dict)
        print('成功加载export最新模型')
    except:
        print('未加载最新模型')
