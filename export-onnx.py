import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import onnx
import torch
import torch.nn as nn
# import torch.ao.quantization.quantize_dynamic as torch_quantize_dynamic
from onnxruntime.quantization import QuantType, quantize_dynamic

from utils import (AttributeDict, setup_logger)
from onnxsim import simplify
from onnx import numpy_helper, helper
from onnxconverter_common import float16
import os

##### usage:
## python3 export-onnx.py --exp_dir ../output --batch 22500

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir",
                        default="onnx_output",
                        type=str,
                        # required=True,
                        help="The experiment dir contains .pt")
    parser.add_argument("--max_seq_length",
                        default=200,
                        type=int,
                        # required=True,
                        help="The sequence length of one sample after SentencePiece tokenization")
    parser.add_argument("--batch",
                        default=-1,
                        type=int,
                        # required=True,
                        help="The batch pt used for decoding")

    return parser

def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)

def export_model(
	model: nn.Module,
	filename: str,
    max_seq_length, int = 200,
	opset_version: int = 11,
) -> None:
    token_ids = torch.ones(1, max_seq_length, dtype=torch.int32)
    valid_ids = torch.ones(1, max_seq_length, dtype=torch.int32)
    label_lens = torch.tensor([200], dtype=torch.int32)

    model = torch.jit.trace(model, (token_ids, valid_ids, label_lens))

    torch.onnx.export(
                model,
				(token_ids, valid_ids, label_lens),
				filename,
                verbose=False,
				opset_version=opset_version,
                # do_constant_folding=False,
				input_names=["token_ids", "valid_ids", "label_lens"],
				output_names=["active_punct_logits"],
				dynamic_axes={
					"token_ids": {0: "N", 1: "T"},
					"valid_ids": {0: "N", 1: "T"},
					"label_lens": {0: "N"},
					"active_punct_logits": {0: "Valid token ids num", 1: "punct num"},
				},
    )

@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    if os.path.exists(args.exp_dir)==False:
        os.mkdir(args.exp_dir)

    from model import punc_model
    model = punc_model()

    checkpoint = torch.load("pytorch_model.bin", map_location="cpu")
    del checkpoint['model.decoder_case.weight']
    del checkpoint['model.decoder_case.bias']
    model.load_state_dict(checkpoint)

    model.to("cpu")
    model.eval()

    opset_version = 13

    logging.info("Exporting model")
    model_filename = args.exp_dir / f"model.onnx"
    export_model(
    	model, 
    	model_filename,
        max_seq_length = args.max_seq_length,
    	opset_version = opset_version,
    )
    logging.info(f"Exported model to {model_filename}")


    onnx_model = onnx.load(model_filename)
    model_sim, check = simplify(onnx_model)
    model_sim_filename = args.exp_dir / f"model_sim.onnx"
    onnx.save(model_sim, model_sim_filename)
    logging.info(f"Exported simplified model to {model_sim_filename}")

    model_filename_int8 = args.exp_dir / f"model.int8.onnx"
    quantize_dynamic(
        model_input=model_sim_filename,
        model_output=model_filename_int8,
        weight_type=QuantType.QUInt8,
    )
    logging.info(f"Exported quantized model to {model_filename_int8}")

    # model_fp16_filename = params.exp_dir / f"model_fp16.onnx"
    # model = onnx.load(model_sim_filename)
    # model_fp16 = float16.convert_float_to_float16(model)
    # onnx.save(model_fp16, model_fp16_filename)

    print('导出onnx成功')



if __name__ == "__main__":
    main()