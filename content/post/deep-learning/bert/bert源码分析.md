---
title: "BERT 源码分析"
description:
date: 2022-04-20T17:24:55+08:00
url: /deep-learning/bert
math: false
draft:  false
categories:
    - Deep Learning
---

解读BERT源码

# bert_tokenzier

## BasicTokenizer

这个类完成普通的分词工作，将英文按照空格分开，把中文一个字一个字分开

```python
def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text) # 转unicode编码
    text = self._clean_text(text) # 遍历一遍文本,清除非文字编码,并且固定空格的编码(\n,\t,"_"=> "_")
    text = self._tokenize_chinese_chars(text) # 在中文汉字两两侧都加上空格
    orig_tokens = whitespace_tokenize(text) # 用空格划分文本
    split_tokens = []
    for token in orig_tokens:
        if self.do_lower_case:
            token = token.lower()
            token = self._run_strip_accents(token) # 如果一个token内出现了标点符号，则把这个token按照标点符号划分开
        split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens
```

## WordpieceTokenizer

划分子词，这个类的输入是BasicTokenizer已经做好基本划分的token，使用贪心的最长匹配找出现在词表(vocab)中的词，时间复杂度$O(n^2)$

示例

```
     For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
```

使用两个指针，start和end,初始化start=0，使用end(init end = len(text))去遍历text[start,len(text)]部分，按照逆序(end--)的顺序依次检查text[start,end]是否在词表中，在则可以根据找到的词的长度移动start指针到找到的词之后，开始下一次循环，如果end减到头都没有找到，说明词表没法划分这个词到子词，可以直接退出循环，添加unk_token

```python
    def tokenize(self, text): 
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```

# configuration_bert

bert的设置类,内容比较简单

```python
    def __init__(self,
                 vocab_size_or_config_json_file=30522, # 词表大小
                 hidden_size=768, # 隐藏层大小
                 num_hidden_layers=12, # transformer层数
                 num_attention_heads=12, # 多头注意力的头数
                 intermediate_size=3072, # transformer ffn的线性变化输出的维度
                 hidden_act="gelu", # 使用的激活函数
                 hidden_dropout_prob=0.1, # 非attention部分的dropout概况
                 attention_probs_dropout_prob=0.1, # attention部分的dropout概率
                 max_position_embeddings=512, # 可以使用position_embedding的最大长度
                 type_vocab_size=2, # token_type_ids的个数,最多两句话，因此为2
                 initializer_range=0.02, #和参数初始化，初始化参数的高斯分布的std
                 layer_norm_eps=1e-12, # ln 参数
                 output_attentions=False, # 是否输出attention_score
                 output_hidden_states=False # 是否输出hidden_states
                 ):
        self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
```

# modeling_bert

bert的模型类,先看总体结构

```python
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
		
        self.apply(self._init_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) # 默认的attention_mask是全1
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids) # 默认token_type_ids是全0

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # 形状和self-attention的计算有关

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # 1处为0，0处为很大的负值

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # 创建head_mask,来特定频闭掉某个头的作用
        # head_mask size [num_heads] 或者 [num_hidden_layers,num_heads],特定指定屏蔽掉某层的某头的作用
        # 最后使用的head_mask的size [num_hidden_layers,-1,num_hedas,-1,-1]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
		# BertEmbedding
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        # transformer encoder
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        # pooler
        pooled_output = self.pooler(sequence_output)
		# output
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
```

## BertEmbeddings(nn.Module)

```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

主要内容是三个embedding层(nn.Embedding)

```python
self.word_embeddings, self.position_embeddings, self.token_type_embeddings 
```

这里先回顾一下embedding的作用,看一下ptroch的官方介绍

1.一个简单的查找表，存储embedding的固定词典和大小

2.此模块通常用于存储单词embedding并使用索引来检索他们，模块的输入是一个索引列表，输出是相应的词embedding

所以通过这个类，我们就可以得到word_embedding，position_embedding和token_type_embedding

**forword函数**

输入

- `input_ids`,`token_type_ids`,`position_ids`

- 类内部也提供了默认的绝对的`position_ids`生成方式，从0到seq_length

- 类内部也提供了默认的token_type_ids的生成，全0

  forword函数的主要内容是获取3个embedding,然后相加，之后过LayerNorm和Dropout返回。因此最后的输出大小是`(batch_size,hidden_size)`

## BertEncoder

```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
```

表示transformer的encoder部分，一个BertLayer就是一层encoder

## BertLayer

```python
class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs
```

- BertAttention完成self-attention的计算
- BertIntermediate是transformer的ffn部分
- BertOutput是最后的add & ln部分

###  BertAttention

```python
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config) # part1
        self.output = BertSelfOutput(config) # part2
        self.pruned_heads = set() # 没有使用

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
```

#### BertSelfAttention

```python
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size) # # 用于得到query的Linear
        self.key = nn.Linear(config.hidden_size, self.all_head_size) # # 用于得到key的Linear
        self.value = nn.Linear(config.hidden_size, self.all_head_size) # # 用于得到value的Linear 

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 将多头混在一起的的Linear变换结果分开
        # (batch_size,seq_length,hidden_size) 
        # => (batch_size,seq_length,num_heads,head_size)
        # => (batch_size,num_heads,seq_length,head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # 公式中的 Q * K^T
        # => (batch_size,num_heads,seq_length,seq_length) # 最后一维是一个token /cdots 其他token的结果
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # 除sqrt(d_k)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask # 把不做attention的地方加上了很大的负数

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # 沿head_size方向做softmax

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs) # dropout

        # Mask heads if we want to
        if head_mask is not None: # head_mask size (1,num_heads,1,1),mask之后变成0
            attention_probs = attention_probs * head_mask 
		# *V的过程,即按照得到的attention_probs对value进行加权求和
        context_layer = torch.matmul(attention_probs, value_layer) # => (batch_size,num_heads,seq_length,head_size)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # => (batch_size,seq_length,num_heads,head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # 拼接多头的输出
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
```

按照自注意力的公式计算
$$
output = softmax(\frac{Q K^T}{\sqrt{d_k}},dim = -1)V \\
Q\_size,K\_{size},V\_{size} = [N,seq\_{length},hidden\_{size}]
$$
多头的处理

只使用1一个Linear，将得到的结果按照头数均分，比如768维，12头，那么每一头Linear之后的的大小应该是768/12=64

计算完成之后将每一头的输出拼接起来，从head_size再次变成hidden_size

#### BertSelfOutput

```python
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # input_tensor是attention部分的输入
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

在得到attention的输出之后，完成add & ln的部分

### BertIntermediate

```
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```

完成transformer的ffn的部分，由linear+actfun组成

### Bertoutput

```
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

在ffn之后，在将维度变换回去，并且完成ffn之后的add & ln部分