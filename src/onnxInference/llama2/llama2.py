# Llama 2 for ONNX
# Modified to work with ONNX and ONNXRuntime

from typing import Optional
from dataclasses import dataclass

import math
import torch

# from helper import random_weight

import os


# save_dir = os.path.join(os.path.expanduser('~'), 'IPU', 'gen_ai', 'llama2', 'model_debug')

@dataclass
class ModelArgs:
    dim: int = 4096  # embedding size
    n_layers: int = 32  # number of stacked transformer decoder blocks
    n_heads: int = 32  # number of heads for queries
    n_kv_heads: int = 32  # number of heads for keys and values
    vocab_size: int = 32000  # defined later by tokenizer
    ffn_dim_multiplier: Optional[float] = 1.0  # set to 1.0 for llama 2 7b and 13b models
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-06  # to avoid division-by-zero in RMSNorm

    hidden_dim: int = 11008  # 14336 for mistral 7b
    # window_size: int = 4096

    batch_size: int = 6  # batch size = 1 for inference
    max_seq_len: int = 2048


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_heads (int): Number of query heads.
            n_repeat (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.num_repeat = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // self.n_heads

        self.wq = torch.nn.Linear(
            args.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )

        # self.wq.weight = random_weight(size=(args.dim, self.n_heads * self.head_dim))

        self.wk = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )

        # self.wk.weight = random_weight(size=(args.dim, self.n_kv_heads * self.head_dim))

        self.wv = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )

        # self.wv.weight = random_weight(size=(args.dim, self.n_kv_heads * self.head_dim))

        self.wo = torch.nn.Linear(
            self.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        # self.wo.weight = random_weight(size=(self.n_heads * self.head_dim, args.dim))

        # torch.save(self.wq, os.path.join(save_dir, 'wq.pt'))
        # torch.save(self.wk, os.path.join(save_dir, 'wk.pt'))
        # torch.save(self.wv, os.path.join(save_dir, 'wv.pt'))
        # torch.save(self.wo, os.path.join(save_dir, 'wo.pt'))

        # KV cache

        self.register_buffer('cache_k', torch.zeros((args.batch_size, self.n_kv_heads, 0, self.head_dim)))       
        self.register_buffer('cache_v', torch.zeros((args.batch_size, self.n_kv_heads, 0, self.head_dim)))

    def reshape_for_broadcast(
            self,
            freqs_ten: torch.Tensor
    ) -> torch.Tensor:
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        Broadcast freq_ten from shape (a, b) to (1, 1, a, b)
        
        """
        return freqs_ten.unsqueeze(0).unsqueeze(1)

    def apply_rotary_emb(
            self,
            x_ten: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor.
    
        This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
        frequency tensors 'freqs_cos' and 'freqs_sin'. The input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
        returned as real tensors.
    
        Args:
            x_ten (torch.Tensor): Tensor to apply rotary embeddings.
            freqs_cos (torch.Tensor): Precomputed frequency tensor for complex exponentials.
            freqs_sin (torch.Tensor): Precomputed frequency tensor for complex exponentials.
    
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    
        """

        # xq := (batch_size, n_heads, seq_len, dim // n_heads)
        # xq_* := (batch_size, n_heads, seq_len, dim // (n_heads * 2))
        x_r, x_i = x_ten.reshape(*x_ten.size()[:-1], -1, 2).unbind(-1)

        # freqs_* := (seq_len, dim // (n_heads * 2)) -> (1, 1, seq_len, dim // (n_heads * 2))
        freqs_cos, freqs_sin = self.reshape_for_broadcast(freqs_cos), self.reshape_for_broadcast(freqs_sin)

        # perform 'complex' multiplication 
        x_out_r = x_r * freqs_cos - x_i * freqs_sin
        x_out_i = x_r * freqs_sin + x_i * freqs_cos

        # x_out := (batch_size, n_heads, seq_len, dim // n_heads)
        x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(x_ten.ndim - 1, end_dim=-1)

        return x_out

    # not required for llama 2 7b and 13b models
    def repeat_kv(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return torch.repeat_interleave(x, dim=dim, repeats=self.num_repeat)

    def memory_efficient_attention(
            self,
            q_i: torch.Tensor,
            k_i: torch.Tensor,
            v_i: torch.Tensor
    ):
        """

        Parameters
        ----------
        q_i : torch.Tensor
            query
        k_i : torch.Tensor
            key
        v_i : torch.Tensor
            value

        Returns
        -------
        None.

        """

        e_si = torch.exp(q_i * k_i)

        v = v_i * torch.exp(q_i * k_i)
        s = e_si

        return NotImplementedError

    def forward(
            self,
            x: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.tensor,
            # cache_k: torch.Tensor,
            # cache_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cos and freqs_sin (torch.Tensor): Precomputed frequency tensors.
            cache_k (tensor.Tensor): Keys cache. For avoiding mutation assignment.
            cache_v (tensor.Tensor): Values cache. For avoiding mutation assignment.
            mask (torch.Tensor, optional): Attention mask tensor. => Removed for inference

        Returns:
            torch.Tensor: Output tensor after attention.
            torch.Tensor: Keys
            torch.Tensor: Values

        """
        bsz, seqlen, _ = x.size()

        # torch.save(x, os.path.join(save_dir, 'x.pt'))

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        # torch.save(xq, os.path.join(save_dir, 'xq.pt'))
        # torch.save(xk, os.path.join(save_dir, 'xk.pt'))

        xq = self.apply_rotary_emb(xq, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        xk = self.apply_rotary_emb(xk, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        # torch.save(xq, os.path.join(save_dir, 'rotary_emb_xq.pt'))
        # torch.save(xk, os.path.join(save_dir, 'rotary_emb_xk.pt'))

        # KV cache
        keys = torch.cat((self.cache_k, xk), dim=2)  # (bs, n_kv_heads, cache_len + seqlen, head_dim)
        values = torch.cat((self.cache_v, xv), dim=2)  # (bs, n_kv_heads, cache_len + seqlen, head_dim)

        # performing attention: keys and values with repeat k/v heads
        # scores := (bs, n_heads, seqlen, cache_len)
        # keys := (bs, n_kv_heads, cache_len + seqlen, head_dim)
        # values := (bs, n_kv_heads, cache_len + seqlen, head_dim)
        keys = self.repeat_kv(keys, dim=1)
        values = self.repeat_kv(values, dim=1)

        # torch.save(keys, os.path.join(save_dir, 'keys.pt'))
        # torch.save(values, os.path.join(save_dir, 'values.pt'))

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim)  # torch.sqrt(torch.tensor([self.head_dim]))
        scores = torch.nn.functional.softmax(scores, dim=-1)

        # torch.save(scores, os.path.join(save_dir, 'scores.pt'))
        # torch.save(scores, os.path.join(save_dir, 'scores_mask.pt'))        
        # torch.save(scores, os.path.join(save_dir, 'scores_sm.pt'))

        # values := (bs, n_kv_heads, cache_len, head_dim)
        # output := (bs, n_heads, seqlen, head_dim)
        output = torch.matmul(scores, values)  # self.repeat_kv(values, dim=1))
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # torch.save(output, os.path.join(save_dir, 'attention_out.pt'))

        return self.wo(output)


class FeedForwardNetwork(torch.nn.Module):
    """
    position-wise feed forward neural network
    """

    def __init__(self, args: ModelArgs):
        super(FeedForwardNetwork, self).__init__()

        # self.hidden_dim = int((8 * args.dim * args.ffn_dim_multiplier) / 3)
        # self.hidden_dim = args.multiple_of * ((self.hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = torch.nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False,
        )

        # self.w1.weight = random_weight(size=(args.hidden_dim, args.dim))

        self.w2 = torch.nn.Linear(
            args.hidden_dim,
            args.dim,
            bias=False,
        )

        # self.w2.weight = random_weight(size=(args.dim, args.hidden_dim))

        # for SiLU
        self.w3 = torch.nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False,
        )

        # self.w3.weight = random_weight(size=(args.hidden_dim, args.dim))

        # torch.save(self.w1, os.path.join(save_dir, 'w1.pt'))
        # torch.save(self.w2, os.path.join(save_dir, 'w2.pt'))
        # torch.save(self.w3, os.path.join(save_dir, 'w3.pt'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-06):
        super(RMSNorm, self).__init__()

        self.eps = eps

        # gamma parameter
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x) -> torch.Tensor:
        # x := (batch_size, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # should implictly type-cast to float

    def forward(self, x) -> torch.Tensor:
        return self._norm(x) * self.weight


class DecoderBlock(torch.nn.Module):
    def __init__(
            self,
            # layer_id: int,
            args: ModelArgs
    ):
        """
        Initialize a Transformer Decoder Block.

        Args:
            layer_id (int): Identifier for the layer. => Removed to avoid int in input; maybe considered static during graph conversion
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super(DecoderBlock, self).__init__()

        # self.layer_id = layer_id

        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.attention = MultiHeadAttention(args)

        self.feed_forward = FeedForwardNetwork(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # torch.save(self.attention_norm.weight, os.path.join(save_dir, 'attn_norm_wt.pt'))

        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # torch.save(self.ffn_norm.weight, os.path.join(save_dir, 'ffn_norm_wt.pt'))

    def forward(
            self,
            x: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
            # cache_k: torch.Tensor,
            # cache_v: torch.Tensor,
            mask: Optional[torch.Tensor] = None  # omitted since mask is not required for inference
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cos, freqs_sin (torch.Tensor): Precomputed cos and sin frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """

        # torch.save(x, os.path.join(save_dir, 'x_attn.pt'))

        _out = self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin # , cache_k, cache_v
        )

        h = x + _out

        # torch.save(h, os.path.join(save_dir, 'attention.pt'))

        out = h + self.feed_forward.forward(self.ffn_norm(h))

        # torch.save(out, os.path.join(save_dir, 'attention_ffn_out.pt'))

        return out # , cache_k, cache_v


class Transformer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (nn.Embedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (nn.Linear): Linear layer for final output.
            freqs_cos, freqs_sin (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super(Transformer, self).__init__()

        self.params = args  # model params
        self.head_dim = args.dim // args.n_heads

        # Embedding Matrix := convert input ids to embedding vectors
        # embedding vectors encode the meaning of the word, trained along with model,
        # similar words have their embedding vector closer in distance
        self.tok_embeddings = torch.nn.Embedding(
            args.vocab_size, args.dim,
        )

        # torch.save(self.tok_embeddings, os.path.join(save_dir, 'tok_emb.pt'))

        # self.tok_embeddings.weight = random_weight(size=(args.vocab_size, args.dim))

        # Model layers
        self.layers = torch.nn.ModuleList()

        for i in range(args.n_layers):
            self.layers.append(DecoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = torch.nn.Linear(
            args.dim, args.vocab_size, bias=False,
        )

        # torch.save(self.output, os.path.join(save_dir, 'ffn_lin_out.pt'))

        # self.output.weight = random_weight(size=(args.vocab_size, args.dim))

        freqs_cos, freqs_sin = self.precompute_freqs(args.dim // args.n_heads, args.max_seq_len * 2)

        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        # self.register_buffer('k', torch.zeros((self.params.batch_size, self.params.n_kv_heads, 0, self.params.dim // self.params.n_heads)))
        # self.register_buffer('v', torch.zeros((self.params.batch_size, self.params.n_kv_heads, 0, self.params.dim // self.params.n_heads)))

        # position of token in sentence/prompt
        self.register_buffer('token_pos', torch.tensor(0))

    def precompute_freqs(
            self,
            size: int,
            seq_len: int,
            theta: float = 10000.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'size'
        and the end index 'seq_len'. The 'theta' parameter scales the frequencies.
        The returned tensor contains complex values in complex64 data type.

        Args:
            size (int): Dimension of the frequency tensor.
            seq_len (int): Seq Len.
            theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

        Returns:
            [torch.Tensor, torch.Tensor]: Precomputed frequency tensor with real exponentials: one for cos and the other for sin.
          
        """
        # shape := (dim // (n_heads * 2),)
        freqs = 1.0 / (theta ** (torch.arange(0, size, step=2) / size))  # should implictly type-cast to float

        # shape := (seq_len,)
        m = torch.arange(seq_len)

        # shape := (seq_len, dim // (n_heads * 2))
        freqs = torch.outer(m, freqs)

        # torch.save(freqs, os.path.join(save_dir, 'freqs.pt'))

        freqs_cos, freqs_sin = torch.cos(freqs), torch.sin(freqs)

        return freqs_cos, freqs_sin

    @torch.inference_mode()
    def forward(
            self,
            tokens: torch.Tensor,
            # indices: torch.Tensor,
            # keys: tuple[torch.Tensor],
            # values: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor], tuple[torch.Tensor]]:
        """
        Perform a forward pass through the Transformer model.
        
        Attention Mask omitted since it is not required for inference

        Args:
            tokens (torch.Tensor): Input token indices.
            indices (torch.Tensor): Indices for selecting freqs_cos and freqs_sin values. Passed as input since addition of seqlen counts as mutation.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """

        _bsz, seqlen = tokens.size()

        h = self.tok_embeddings(tokens)

        # torch.save(tokens, os.path.join(save_dir, 'tok.pt'))
        # torch.save(h, os.path.join(save_dir, 'h.pt'))
        # torch.save(self.freqs_cis, os.path.join(save_dir, 'self_freqs_cis.pt'))

        indices = torch.arange(self.token_pos, self.token_pos + seqlen, dtype=torch.int)  
        freqs_cos = torch.index_select(self.freqs_cos, 0, indices)
        freqs_sin = torch.index_select(self.freqs_sin, 0, indices)

        # torch.save(torch.complex(freqs_cos, freqs_sin), os.path.join(save_dir, 'freqs_cis.pt'))

        # KV cache for next input token phrase
        # cache_k, cache_v = tuple(), tuple()

        self.token_pos += seqlen

        for i, layer in enumerate(self.layers):
            # torch.save(h, os.path.join(save_dir, 'h_' + str(i) + '.pt'))

            h = layer(h, freqs_cos, freqs_sin) # , keys[i], values[i])

            # torch.save(h, os.path.join(save_dir, 'llama2_' + str(i) + '.pt'))

            # update KV cache with individual kv-caches for each layer
            # cache_k += (k,)
            # cache_v += (v,)

        h = self.norm(h)
        out = self.output(h)

        return out # , cache_k, cache_v


if __name__ == '__main__':
    args = ModelArgs()

    decoder = Transformer(args)

    input_tokens = torch.tensor([[1, 23, 33, 454, 333]])
    input_len = 5
    indices = torch.arange(0, input_len, dtype=torch.int)
    cache_k = [torch.zeros((args.batch_size, args.n_kv_heads, 0, args.dim // args.n_heads))] * args.n_layers
    cache_v = [torch.zeros((args.batch_size, args.n_kv_heads, 0, args.dim // args.n_heads))] * args.n_layers

    out, keys, values = decoder(input_tokens) # , indices, cache_k, cache_v)

    print(out.size())
