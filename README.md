# 标点恢复
该项目是在Edge-Punct-Casing基础上进行修改，主要修改为：<br>
1.BPE模型修改为sensevoice项目的BPE模型，并增加中英文标点符号共61种。 <br>
2.使用deepctrl-sft-data数据集进行训练。 <br>
3.删除英文大小写预测。 <br>

# 运行代码
python test.py

# 局限性
1.由于训练数据中文偏多，英文偏少，所以英文标点预测效果较差。 <br>
2.训练数据不足，预测效果有待提升。 <br>
3.代码和能力有限。 <br>

# 代码和数据链接
sensevoice:https://github.com/FunAudioLLM/SenseVoice <br>
Edge-Punct-Casing：https://github.com/frankyoujian/Edge-Punct-Casing <br>
deepctrl-sft-data：https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data <br>