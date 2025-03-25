import sentencepiece as spm
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
punc_list = [
    "々", "!", "\"", "#", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", 
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", 
    "·", "—", "―", "‘", "’", "“", "”", "…", "‧", "℃", "○", "、", "，", "。",
    "！", "？", "：", "；", "〃", "〆", "〇", "〈", "〉", "《", "》", "「", "」", 
    "『", "』", "【", "】", "〒", "〜", "〝", "〞", "（", "）", "｛", "｝"
]

punctuation_map_decode={v+1:k for v,k in enumerate(punc_list)}
punctuation_map_decode[0]=''


def log_softmax(x, axis=-1):
    """NumPy 实现的 log_softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)  # 防止溢出，稳定计算
    x_exp = np.exp(x - x_max)
    return np.log(x_exp / np.sum(x_exp, axis=axis, keepdims=True))

def pre_handle(text,
    tokenizer: spm.SentencePieceProcessor, 
    max_seq_length: int = 200,
):
    # 将 token 转换为 numpy 数组，并加上 <s> 和 </s> 标记
    word_tokens = np.array([1] + tokenizer.encode(text, out_type=int) + [2], dtype=np.int32)
    
    # 将 token 转换为字符串形式，并加上 <s> 和 </s> 标记
    tokens_str = ['<s>'] + tokenizer.encode(text, out_type=str) + ['</s>']
    
    # 填充数组到 max_seq_length 长度
    padded_word_tokens = np.pad(word_tokens, (0, max_seq_length - word_tokens.size), mode='constant', constant_values=0)
    
    # 计算 valid_ids，0 的位置标记为无效
    valid_ids = (padded_word_tokens != 0).astype(np.int32)
    
    # 计算有效标记的数量（即 valid_ids 中的 1 的总数）
    label_len = np.sum(valid_ids)
    
    # 使用 np.expand_dims 扩展维度，模拟 unsqueeze 的功能
    return tokens_str, np.expand_dims(padded_word_tokens, axis=0).astype(np.int32), np.expand_dims(valid_ids, axis=0).astype(np.int32), np.expand_dims(label_len, axis=0).astype(np.int32)

# 加载模型
sp = spm.SentencePieceProcessor()
sp.Load('chn_jpn_yue_eng_ko_spectok.bpe.model')

import onnx
import onnxruntime as ort
from onnx import numpy_helper, helper

class OnnxModel:
	def __init__(self, model_filename: str):
		session_opts = ort.SessionOptions()
		session_opts.inter_op_num_threads = 1
		session_opts.intra_op_num_threads = 1
		# session_opts.use_deterministic_compute = 1

		self.session_opts = session_opts

		self.init_model(model_filename)

	def init_model(self, model_filename: str):
		self.model = ort.InferenceSession(
			model_filename,
			sess_options = self.session_opts,
			providers = ["CPUExecutionProvider"],
		)

	def run_model(
		self, 
		token_ids: np.ndarray, 
		valid_ids: np.ndarray, 
		label_lens: np.ndarray,
	) -> Tuple[np.ndarray, np.ndarray]:

		out = self.model.run(
			[
				self.model.get_outputs()[0].name,
			],
			{
				self.model.get_inputs()[0].name: token_ids,
				self.model.get_inputs()[1].name: valid_ids,
				self.model.get_inputs()[2].name: label_lens,
			},
		)

		return out[0]

model = OnnxModel(model_filename = 'onnx_output/model.onnx')

text_list=['你好what is your name哈哈',
           '你叫什么名字好的我知道了',
           '我很高兴',
           'what is your name can you help me',
           'thank your very much',
           '我们都是木头人不会讲话不会动',
           '你知道道德经这本书吗',
           '测试一下标点恢复的效果你今天去哪里玩啊今天去钓鱼',
           '每种文本类型都根据其应用场景和目标受众精心调整了风格通过这种设计Cosmopedia不仅能应用于学术研究还能广泛应用于教育娱乐和技术等领域',
           'FunASR希望在语音识别的学术研究和工业应用之间架起一座桥梁通过发布工业级语音识别模型的训练和微调研究人员和开发人员可以更方便地进行语音识别模型的研究和生产并推动语音识别生态的发展让语音识别更有趣',
           ]

# 这里是逐个推理，支持batch推理
for text in text_list:

    token_strs,token_ids,valid_ids,label_lens = pre_handle(text,sp,max_seq_length=200)

    active_punct_logits = model.run_model(token_ids, valid_ids=valid_ids, label_lens=label_lens)

    punct_pred = np.argmax(log_softmax(active_punct_logits, axis=1), axis=1)

    punct_str_list=[punctuation_map_decode[i] for i in punct_pred.tolist()]

    print(text,'---加标点结果为：')

    new_text=''
    for k,v in zip(token_strs[1:-1],punct_str_list):
        # print(k+v,end='')
        new_text+=k+v
    print(new_text.replace('▁',' '),'\n'*2)
