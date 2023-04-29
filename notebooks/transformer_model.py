import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================

        mean = inputs.mean(dim=-1, keepdim=True)
        var = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)
        outputs = (inputs - mean) / torch.sqrt(var + self.eps)
        outputs = self.weight*outputs + self.bias
        return outputs

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        # ==========================
        # TODO: Write your code here
        # ==========================
        self.qkv_weights = nn.Linear(self.num_heads*self.head_size, 3*self.num_heads*self.head_size)
        # self.qkv_weights = torch.one
        self.w_o = nn.Linear(self.num_heads*self.head_size, self.num_heads*self.head_size)


    def get_attention_weights(self, queries: torch.Tensor, keys: torch.Tensor, mask=None):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 
           
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        head_size = queries.shape[-1]
        num_heads = queries.shape[1]
        seq_len = queries.shape[2]

        attn_weights = queries @ keys.swapaxes(2, 3)   # keys = B, N, S, D --> B, N, D, S then batch wise matrix multiplication
        attn_weights = attn_weights/ head_size**0.5

        if mask is not None:
            reshaped_mask = mask.repeat(1, seq_len).reshape(-1, seq_len, seq_len)   # B, S --> B, S, S
            reshaped_mask = reshaped_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)   # B, S, S --> B, N, S, S
            attn_weights = attn_weights.masked_fill(reshaped_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        return attn_weights
        
    def apply_attention(self, queries, keys, values, mask=None):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. 

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. 
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. 
        """

        # ==========================
        # TODO: Write your code here
        # ==========================

        attn_weights = self.get_attention_weights(queries, keys, mask=mask)     # B, N, S, S
        h = attn_weights @ values           # values = B, N, S, D
        h = self.merge_heads(h)
        return h

    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        b, s, nd = tensor.shape
        head_size = int(nd / self.num_heads)
        output = tensor.reshape(b, s, self.num_heads, head_size)   # B, S, ND --> B, S, N, D
        output = output.swapaxes(2, 1)          # B, N, S, D
        return output
        
    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        b, n, s, d = tensor.shape
        output = tensor.swapaxes(1, 2)          # B, N, S, D --> B, S, N, D
        output = output.reshape(b, s, n*d)
        return output

    def forward(self, hidden_states, mask=None):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.
            
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================

        qkv = self.qkv_weights(hidden_states)      # [B, S, ND] --> [B, S, 3*ND]: paralleize qkv weight matrix projection
        chunk_size = qkv.shape[-1] // 3
        q, k, v = qkv.split(split_size=chunk_size, dim=-1)  # [B, S, 3ND] -->  [B, S, ND] : split in q, k, v
        q = q.reshape(-1, q.shape[1], self.num_heads, self.head_size)
        q = q.swapaxes(2, 1)
        k = k.reshape(-1, k.shape[1], self.num_heads, self.head_size)
        k = k.swapaxes(2, 1)
        v = v.reshape(-1, v.shape[1], self.num_heads, self.head_size)
        v = v.swapaxes(2, 1)

        # q,k,v = hidden_states

        v_attn = self.apply_attention(q, k, v, mask=mask)
        v_attn = self.w_o(v_attn)           # [B, S, ND]
        return v_attn


class PostNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, sequence_length, dropout=0.30):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attention_outputs = self.attn(x, mask)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs = self.linear(attention_outputs)

        outputs = self.layer_norm_2(outputs + attention_outputs)
        return outputs


class PreNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads,sequence_length, dropout=0.0):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.

        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            sequence_length - Length of the sequence
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads, sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # ==========================
        # TODO: Write your code here
        # ==========================

        inp = self.layer_norm_1(x)
        inp = self.attn(inp, mask=mask)
        inp += x

        out = self.layer_norm_2(inp)
        out = self.linear(out)
        out += inp

        return out


class Transformer(nn.Module):
    
    def __init__(self, embed_dim=256, hidden_dim=256, num_heads=1,
            num_layers=2, block='prenorm', dropout=0.3):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            block - Type of attention block
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
        """
        super().__init__()
        
        #Adding the cls token to the sequnence 
        self.sequence_length = 136
        # Layers/Networks
        if block =='prenorm':
          self.transformer = nn.ModuleList([PreNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        else:
          self.transformer = nn.ModuleList([PostNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings)
        self.pos_embedding = nn.Parameter(torch.randn(1,self.sequence_length,embed_dim))
   
    def forward(self, x, mask=None, labels=None):
        """Transformer

        This is a small version of  Transformer

        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, sequence length)`)
            The input tensor containing text.
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, embed_dim)`)
            A tensor containing the output from the mlp_head.
        """
        # Preprocess input

        B, T, _ = x.shape

        # Add CLS token and positional encoding
        x = x + self.pos_embedding

        #Add dropout and then the transformer
        # ==========================
        # TODO: Write your code here
        # ==========================

        # if mask is not None:
        #     mask = torch.cat((torch.on.cuda(), mask), dim=1)

        x = self.dropout(x)
        for block in self.transformer:
            x = block(x, mask)

        #Take the cls token representation and send it to mlp_head
 
        # ==========================
        # TODO: Write your code here
        # ==========================

        x = self.dropout(x)
        x = self.mlp_head(x)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fn(x.swapaxes(1,2), labels)
            return loss, x
        return x
