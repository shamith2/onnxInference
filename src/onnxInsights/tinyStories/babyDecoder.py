# babyLlama
# Adapted from https://github.com/meta-llama/llama/blob/main/llama/model.py

from typing import Optional
from dataclasses import dataclass

import math
import torch


@dataclass
class ModelArgs:
    dim: int = 1024  # embedding size
    n_layers: int = 4  # number of stacked transformer decoder blocks
    n_heads: int = 8  # number of heads for queries, keys and values
    vocab_size: int = -1  # defined later by tokenizer
    norm_eps: float = 1e-06  # to avoid division-by-zero in RMSNorm

    # include 'multiple_of: make SwiGLU hidden layer size multiple of large power of 2',
    # if any, in hidden_dim
    hidden_dim: int = 768

    device: str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_batch_size: int = 8
    max_seq_len: int = 1024


class MultiHeadAttention(torch.nn.Module):
    """Multi-Head Self Attention module"""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Multi-Head Self Attention module

        Args:
            args (ModelArgs): Model configuration parameters

        Attributes:
            n_heads (int): Number of heads
            head_dim (int): Dimension size of each attention head
            wq (nn.Linear): Linear transformation for queries
            wk (nn.Linear): Linear transformation for keys
            wv (nn.Linear): Linear transformation for values
            wo (nn.Linear): Linear transformation for output

        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = args.n_heads
        self.head_dim = args.dim // self.n_heads

        self.wq = torch.nn.Linear(
            args.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )

        self.wk = torch.nn.Linear(
            args.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )

        self.wv = torch.nn.Linear(
            args.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )

        self.wo = torch.nn.Linear(
            self.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        # KV cache
        # self.register_buffer('cache_k', torch.zeros((args.max_batch_size, self.n_heads, 0, self.head_dim),
        #                                             device=args.device), persistent=False)    
        # self.register_buffer('cache_v', torch.zeros((args.max_batch_size, self.n_heads, 0, self.head_dim),
        #                                             device=args.device), persistent=False)

    def forward(
            self,
            in_token: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the multi-head self attention module

        Args:
            in_token (torch.Tensor): Input token tensor
            mask (torch.Tensor, optional): Attention mask tensor

        Returns:
            torch.Tensor: Output tensor after attention

        """
        bsz, seqlen, _ = in_token.size()

        x_queries, x_keys, x_values = self.wq(in_token), self.wk(in_token), self.wv(in_token)

        x_queries = x_queries.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        x_keys = x_keys.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        x_values = x_values.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # KV cache
        # keys = torch.cat((self.cache_k, x_keys), dim=2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        # values = torch.cat((self.cache_v, x_values), dim=2)  # (bs, n_heads, cache_len + seqlen, head_dim)

        scores = torch.matmul(x_queries, x_keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # for inference, mask is None
        if torch.is_tensor(mask):
            scores = scores + mask

        scores = torch.nn.functional.softmax(scores, dim=-1)

        # values := (bs, seqlen, n_heads, head_dim)
        # output := (bs, seqlen, n_heads, head_dim)
        output = torch.matmul(scores, x_values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForwardNetwork(torch.nn.Module):
    """
    Position-wise feed forward neural network
    """

    def __init__(self, args: ModelArgs):
        super(FeedForwardNetwork, self).__init__()

        self.w1 = torch.nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False,
        )

        self.w2 = torch.nn.Linear(
            args.hidden_dim,
            args.dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.relu(self.w1(x)))


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

        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cos, freqs_sin (torch.Tensor): Precomputed cos and sin frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        _out = self.attention.forward(
            self.attention_norm(x), mask
        )

        h = x + _out

        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out


class babyDecoder(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Initialize babyDecoder model.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (nn.Embedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (nn.Linear): Linear layer for final output.

        """
        super(babyDecoder, self).__init__()

        self.params = args  # model params
        self.device = args.device
        self.head_dim = args.dim // args.n_heads
        self.max_seq_len = args.max_seq_len

        # Embedding Matrix := convert input ids to embedding vectors
        # embedding vectors encode the meaning of the word, trained along with model,
        # similar words have their embedding vector closer in distance
        self.tok_embedding = torch.nn.Embedding(
            args.vocab_size, args.dim,
        )

        self.pos_embedding = torch.nn.Embedding(
            args.max_seq_len, args.dim
        )

        # Model layers
        self.layers = torch.nn.ModuleList()

        for i in range(args.n_layers):
            self.layers.append(DecoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = torch.nn.Linear(
            args.dim, args.vocab_size, bias=False,
        )
    
    def sample_top_p(
            self,
            probs: torch.Tensor,
            p: float = 0.95
    ) -> torch.Tensor:
        """
        Perform top-p (nucleus) sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.

        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        
        return next_token

    @torch.inference_mode()
    def generate(
            self,
            tokens: torch.Tensor,
            temperature: float = 1.0,
            top_p: float = 0.99,
            eos_token_id: int = 2
    ) -> torch.Tensor:
        """
        Perform an inference forward pass through the babyLlama model
        Attention Mask omitted since it is not required for inference

        Args:
            tokens (torch.Tensor): Input token indices
        
        Returns:
            torch.Tensor: Output tokens after inference
        
        """

        for cur_pos in range(len(tokens), self.max_seq_len):
            logits = self(tokens)

            if temperature > 0:
                probs = torch.nn.functional.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(0)
                
            tokens = torch.cat((tokens, next_token), dim=-1)

            # break if end of sentence token is generated
            if next_token.item() == eos_token_id:
                break
        
        return tokens

    def forward(
            self,
            tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a forward pass through the babyDecoder model

        Args:
            tokens (torch.Tensor): Input token indices

        Returns:
            torch.Tensor: Output logits after applying the babyDecoder model

        """

        _bsz, seqlen = tokens.shape
        h = self.tok_embedding(tokens) + self.pos_embedding(torch.arange(seqlen, device=self.device))

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), -torch.inf, device=self.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i
            # mask = torch.hstack([
            #     torch.zeros((seqlen, torch.select(indices, 0, 0)), device=self.device),
            #     mask
            # ])

        for i, layer in enumerate(self.layers):
            h = layer(h, mask)

        h = self.norm(h)
        out = self.output(h)

        return out

if __name__ == '__main__':
    args = ModelArgs()
    args.vocab_size = 2048

    model = babyDecoder(args).to(args.device)

    input_tokens = torch.randint(args.vocab_size, (args.max_batch_size, args.max_seq_len), device=args.device)

    out = model(input_tokens)

    print(out.size())

    print(model)

    num_params = sum(p.numel() for p in model.parameters()) // 1e6
    print('model size: {}M parameters'.format(num_params))

    # from onnx_helper import export_onnx
    
    # input_names = ['tokens']
    # export_onnx(model, 'babyDecoder', 'babyDecoder.onnx', input_names=input_names, model_inputs=input_tokens)
