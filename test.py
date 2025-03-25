import sentencepiece as spm
from tqdm import tqdm
import torch
import torch.nn.functional as F

punc_list = [
    "々", "!", "\"", "#", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", 
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", 
    "·", "—", "―", "‘", "’", "“", "”", "…", "‧", "℃", "○", "、", "，", "。",
    "！", "？", "：", "；", "〃", "〆", "〇", "〈", "〉", "《", "》", "「", "」", 
    "『", "』", "【", "】", "〒", "〜", "〝", "〞", "（", "）", "｛", "｝"
]

punctuation_map_decode={v+1:k for v,k in enumerate(punc_list)}
punctuation_map_decode[0]=''

def pre_handle(text,
    tokenizer:spm.SentencePieceProcessor, 
    max_seq_length: int = 200,
    ):
    word_tokens = torch.tensor([1] + tokenizer.encode(text, out_type=int) + [2], dtype=torch.int32)
    tokens_str = ['<s>'] + tokenizer.encode(text, out_type=str) + ['</s>']
    padded_word_tokens = F.pad(word_tokens, (0, max_seq_length-word_tokens.size(0)), value=0)
    valid_ids = (padded_word_tokens != 0).int()
    label_len = valid_ids.sum()
    return tokens_str,padded_word_tokens.unsqueeze(0),valid_ids.unsqueeze(0),label_len.unsqueeze(0)

# 加载模型
sp = spm.SentencePieceProcessor()
sp.Load('chn_jpn_yue_eng_ko_spectok.bpe.model')

from model import punc_model
model = punc_model()

checkpoint = torch.load("pytorch_model.bin", map_location="cpu")
del checkpoint['model.decoder_case.weight']
del checkpoint['model.decoder_case.bias']
model.load_state_dict(checkpoint)
model.eval()

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

    active_punct_logits, final_mask = model(token_ids, valid_ids=valid_ids, label_lens=label_lens)

    punct_pred = torch.argmax(F.log_softmax(active_punct_logits, dim=1), dim=1)

    punct_str_list=[punctuation_map_decode[i] for i in punct_pred.tolist()]

    print(text,'---加标点结果为：')

    new_text=''
    for k,v in zip(token_strs[1:-1],punct_str_list):
        # print(k+v,end='')
        new_text+=k+v
    print(new_text.replace('▁',' '),'\n'*2)
