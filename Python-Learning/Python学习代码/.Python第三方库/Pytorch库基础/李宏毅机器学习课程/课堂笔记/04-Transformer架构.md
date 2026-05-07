Transfomer 是一个 Sequence-to-Sequence(Seq2Seq) 的模型

Seq2Seq 核心特征是：输入一个序列，输出一个序列，并且输出的长度由机器自己决定，Seq2Seq 的模型能解决许多问题，例如：

- 机器翻译 (Machine Translation)
- 语音辨识 (Speech Recognition)
- 语音翻译 (Speech Translation)

- 聊天机器人 (Chatbot)
- 文本摘要 (Text Summarization)
- 问答系统 (Question Answering, QA)

- 文法剖析 (Syntactic Parsing)
- 多标签分类 (Multi-label Classification)
- 目标检测 (Object Detection)

Seq2Seq模型最基本的结构为：

$$
\text{input sequence} \longrightarrow \text{encoder} \longrightarrow \text{decoder} \longrightarrow \text{output sequence}
$$