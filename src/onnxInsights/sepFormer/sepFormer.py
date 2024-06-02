# Implemented from Attention is all you need in Speech Seperation

import unittest
import functools
import sys
from dataclasses import dataclass

import math
import torch

from ..onnxHelpers import ONNXInference

# seed
torch.manual_seed(43)
torch.cuda.seed_all()

@dataclass
class ModelArgs:
    num_speakers: int = 2

    enc_in_channels: int = 1
    enc_out_channels: int = 256
    enc_kernel_size: int = 16
    enc_stride: int = 8

    chunk_size: int = 250

    mha_dim: int = 256
    mha_n_heads: int = 8

    pe_max_len: int = 20000

    sepformer_n_layers: int = 2
    transformer_n_layers: int = 8

    ffn_hidden_dim: int = 1024
    norm_eps: float = 1e-06

    gn_num_groups: int = 1
    gn_eps: float = 1e-8

    device: str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Encoder(torch.nn.Module):
    """
    Encoder block

    """

    def __init__(
            self,
            args: ModelArgs,
    ):

        super(Encoder, self).__init__()

        self.conv1d = torch.nn.Conv1d(
            in_channels=args.enc_in_channels,
            out_channels=args.enc_out_channels,
            kernel_size=args.enc_kernel_size,
            stride=args.enc_stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode='zeros',
            device=args.device
        )

        self.relu = torch.nn.ReLU(inplace=False)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        # x -> (batch_size, in_channels, T)
        # out -> (batch_size, out_channels, T')
        out = self.relu(self.conv1d(x))

        return out


class Decoder(torch.nn.Module):
    """
    Decoder Block
    """
    def __init__(
            self,
            args: ModelArgs
    ):
        super(Decoder, self).__init__()

        self.conv1d_transp = torch.nn.ConvTranspose1d(
            in_channels=args.enc_out_channels,
            out_channels=args.enc_in_channels,
            kernel_size=args.enc_kernel_size,
            stride=args.enc_stride,
            padding=0,
            output_padding=0,
            groups=1,
            bias=False,
            dilation=1,
            padding_mode='zeros',
            device=args.device
        )
    
    def forward(
            self,
            x: torch.Tensor
    ):
        # x -> (batch_size, out_channels, T')
        # out -> (batch_size, in_channels, T')
        out = self.conv1d_transp(x)

        return out


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Self-Attention module
    """

    def __init__(
            self,
            args: ModelArgs
    ):
        """
        Initialize the Multi-Head Self-Attention module

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

        self.n_heads = args.mha_n_heads
        self.head_dim = args.mha_dim // self.n_heads

        self.wq = torch.nn.Linear(
            args.mha_dim,
            self.n_heads * self.head_dim,
            bias=False,
            device=args.device
        )

        self.wk = torch.nn.Linear(
            args.mha_dim,
            self.n_heads * self.head_dim,
            bias=False,
            device=args.device
        )

        self.wv = torch.nn.Linear(
            args.mha_dim,
            self.n_heads * self.head_dim,
            bias=False,
            device=args.device
        )

        self.wo = torch.nn.Linear(
            self.n_heads * self.head_dim,
            args.mha_dim,
            bias=False,
            device=args.device
        )

    def forward(
            self,
            in_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the multi-head self attention module

        Args:
            in_t (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after attention

        """
        bsz, seqlen, _ = in_t.size()

        x_queries, x_keys, x_values = self.wq(in_t), self.wk(in_t), self.wv(in_t)

        x_queries = x_queries.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        x_keys = x_keys.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        x_values = x_values.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(x_queries, x_keys.transpose(2, 3)) / math.sqrt(self.head_dim)

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

    def __init__(
            self,
            args: ModelArgs
    ):
        super(FeedForwardNetwork, self).__init__()

        self.w1 = torch.nn.Linear(
            args.mha_dim,
            args.ffn_hidden_dim,
            bias=False,
            device=args.device
        )

        self.w2 = torch.nn.Linear(
            args.ffn_hidden_dim,
            args.mha_dim,
            bias=False,
            device=args.device
        )

        self.relu = torch.nn.ReLU(inplace=False) 

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        out = self.w2(self.relu(self.w1(x)))

        return out


class RMSNorm(torch.nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super(RMSNorm, self).__init__()

        # gamma parameter
        self.weight = torch.nn.Parameter(torch.ones(args.mha_dim, layout=torch.strided, device=args.device))

    def _norm(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        # x := (batch_size, seqlen, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + args.norm_eps)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self._norm(x) * self.weight


# Adapted PositionEncoding function
# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinusoidalPositionalEncoding(torch.nn.Module):
    """
    Positional Encoding
    """
    def __init__(
            self,
            args: ModelArgs
    ):
        super(SinusoidalPositionalEncoding, self).__init__()

        position = torch.arange(args.pe_max_len, device=args.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.mha_dim, 2, device=args.device) * (-math.log(10000.0) / args.mha_dim))
        
        pe = torch.zeros(args.pe_max_len, args.mha_dim, layout=torch.strided, device=args.device, requires_grad=False)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe, persistent=True)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        # x = (batch_size, time, channels)
        x = x + self.pe[:, :x.size(1)]
        
        return x.clone().detach()


class TransformerBlock(torch.nn.Module):
    """
    Used in InterTransformer and IntraTransformer
    """
    def __init__(
            self,
            args: ModelArgs
    ):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(args)
        self.ffn = FeedForwardNetwork(args)
        
        self.attention_norm = RMSNorm(args)
        self.ffn_norm = RMSNorm(args)
    
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        _out = self.mha.forward(self.attention_norm(x))

        h = x + _out

        out = h + self.ffn.forward(self.ffn_norm(h))

        return out


class InterTransformer(torch.nn.Module):
    """
    Used in InterTransformer and IntraTransformer
    """
    def __init__(
            self,
            args: ModelArgs
    ):
        super(InterTransformer, self).__init__()

        self.pos_encoding = SinusoidalPositionalEncoding(args)

        # InterTransformer layers
        self.layers = torch.nn.ModuleList()

        for _ in range(args.transformer_n_layers):
            self.layers.append(TransformerBlock(args))
        
        self.norm = RMSNorm(args)
    
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        h = x + self.pos_encoding(x)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        return h


class SepFormerLayer(torch.nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super(SepFormerLayer, self).__init__()

        self.intra_transformer = InterTransformer(args)
        self.inter_transformer = InterTransformer(args)

        self.intra_norm = torch.nn.GroupNorm(
            num_groups=args.gn_num_groups,
            num_channels=args.enc_out_channels,
            eps=args.gn_num_groups,
            affine=True,
            device=args.device
        )

        self.inter_norm = torch.nn.GroupNorm(
            num_groups=args.gn_num_groups,
            num_channels=args.enc_out_channels,
            eps=args.gn_num_groups,
            affine=True,
            device=args.device
        )
    
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        # x = (batch_size, channels, time, chunks)
        bsz, c, t, ck = x.size()

        # in_intra = (bsz, c, t, ck) -> (bsz, ck, t, c) -> (bsz * ck, t, c)
        in_intra = x.permute(0, 3, 2, 1).contiguous().view(bsz * ck, t, c)

        # in_intra = (bsz * ck, t, c) -> (bsz * ck, t, c)
        in_intra = self.intra_transformer(in_intra)

        # in_intra = (bsz * ck, t, c) -> (bsz, c, t, ck)
        in_intra = in_intra.view(bsz, ck, t, c).permute(0, 3, 2, 1).contiguous().view(bsz, c, t, ck)

        # residual: norm and add
        out = x + self.intra_norm(in_intra)

        # in_inter = (bsz, c, t, ck) -> (bsz, t, ck, c) -> (bsz * t, ck, c)
        in_inter = out.permute(0, 2, 3, 1).contiguous().view(bsz * t, ck, c)

        # in_inter = (bsz * t, ck, c) -> (bsz * t, ck, c)
        in_inter = self.inter_transformer(in_inter)

        # in_inter = (bsz * t, ck, c) -> (bsz, c, t, ck)
        in_inter = in_inter.view(bsz, t, ck, c).permute(0, 3, 1, 2).contiguous().view(bsz, c, t, ck)
        
        # residual: norm and add
        out = out + self.inter_norm(in_inter)
        
        return out


class SepFormer(torch.nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super(SepFormer, self).__init__()

        # SepFormer layers
        self.module = torch.nn.Sequential()

        for _ in range(args.sepformer_n_layers):
            self.module.append(SepFormerLayer(args))
    
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        out = self.module(x)

        return out


class MaskNet(torch.nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super(MaskNet, self).__init__()

        self.chunk_size = args.chunk_size
        self.num_speakers = args.num_speakers

        self.linear1 = torch.nn.Linear(
            in_features=args.enc_out_channels,
            out_features=args.enc_out_channels,
            bias=False,
            device=args.device
        )

        self.linear2 = torch.nn.Linear(
            in_features=args.enc_out_channels,
            out_features=self.num_speakers * args.enc_out_channels,
            bias=False,
            device=args.device
        )

        self.prelu = torch.nn.PReLU(
            num_parameters=1,
            init=0.25,
            device=args.device
        )

        self.relu = torch.nn.ReLU(inplace=False)

        self.norm = torch.nn.GroupNorm(
            num_groups=args.gn_num_groups,
            num_channels=args.enc_out_channels,
            eps=args.gn_num_groups,
            affine=True,
            device=args.device
        )

        self.sepformer = SepFormer(args)

        # gated output layer
        self.out_ffn = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=args.enc_out_channels,
                out_features=args.enc_out_channels,
                bias=False,
                device=args.device
            ),
            torch.nn.Tanh()
        )

        self.out_ffn_gate = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=args.enc_out_channels,
                out_features=args.enc_out_channels,
                bias=False,
                device=args.device
            ),
            torch.nn.Sigmoid()
        )

        self.ffn = torch.nn.Linear(
            in_features=args.enc_out_channels,
            out_features=args.enc_out_channels,
            bias=False,
            device=args.device
        )

    
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        # x = (batch_size, channels, time)
        bsz, c, t = x.size()

        # norm + linear: x = (bsz, c, t)
        x = self.norm(x)
        # x = (bsz, t, c)
        x = x.permute(0, 2, 1).contiguous()
        # x = (bsz * t, c)
        x = self.linear1(x.view(bsz * t, c))
        # x = (bsz, c, t)
        x = x.view(bsz, t, c).permute(0, 2, 1).contiguous()

        # chunking: 50% overlapping chunks of size args.chunk_size on time axis
        # x = (bsz, c, chunk_size, num_chunks)

        # method 1:
        # x = x.unfold(dimension=-1, size=self.chunk_size, step=self.chunk_size // 2)
        # x = x.permute(0, 1, 3, 2).contiguous()

        # method 2:
        x = torch.nn.functional.unfold(
            x.unsqueeze(2),
            kernel_size=(1, self.chunk_size),
            dilation=(1, 1),
            padding=(0, self.chunk_size),
            stride=(1, self.chunk_size // 2),
        )

        x = x.view(bsz, c, self.chunk_size, -1)
        _, _, _, chk = x.size()

        # sepformer
        x = self.sepformer.forward(x)

        # prelu + linear
        x = self.prelu(x)

        # x = (bsz, c, chunk_size, num_chunks) -> (bsz, chunk_size, num_chunks, c)
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.linear2(x.view(-1, c))

        # x = (bsz, chunk_size, chk, c * num_speakers) -> (bsz, chunk_size, chk, c, num_speakers)
        # x = (bsz, chunk_size, chk, c, num_speakers) -> (bsz, num_speakers, c, chunk_size, chk)
        x = x.view(bsz, self.chunk_size, chk, c, self.num_speakers)
        x = x.permute(0, 4, 3, 1, 2).contiguous()

        # overlap-add
        x = torch.nn.functional.fold(
            x.view(bsz * self.num_speakers, c * self.chunk_size, chk),
            output_size=(1, t),
            kernel_size=(1, self.chunk_size),
            dilation=(1, 1),
            padding=(0, self.chunk_size),
            stride=(1, self.chunk_size // 2),
        )

        # 2 ffn + relu
        # x = (bsz * num_speakers, c, t)
        x = x.view(bsz * self.num_speakers, c, t).permute(0, 2, 1).contiguous()
        x = x.view(-1, c)
        
        x = self.out_ffn(x) * self.out_ffn_gate(x)
        x = self.ffn(x)

        # x = (bsz * num_speaker, t, c -> bsz, num_speakers, c, t)
        x = x.view(bsz * self.num_speakers, t, c)
        x = x.permute(0, 2, 1).contiguous().view(bsz, self.num_speakers, c, t)
        x = self.relu(x)

        # x = mask for each speaker
        # x = (num_speakers, bsz, c, t)
        x = x.permute(1, 0, 2, 3).contiguous()

        return x


class Model(torch.nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super(Model, self).__init__()

        self.encoder = Encoder(args)
        self.masknet = MaskNet(args)
        self.decoder = Decoder(args)

        self.num_speakers = args.num_speakers
        self.dim = 2
    
    def forward(
            self,
            mix: torch.Tensor
    ) -> torch.Tensor:
        enc_mix = self.encoder(mix)

        masks = self.masknet(enc_mix)

        enc_mix = torch.stack([enc_mix] * self.num_speakers)
        sep_mix = enc_mix * masks

        out = torch.cat(
            [
                self.decoder(sep_mix[i]).unsqueeze(-1)
                for i in range(self.num_speakers)
            ],
            dim=-1,
        )

        # T can change after Encoder and Decoder
        out = torch.nn.functional.pad(
            input=out,
            pad=(0, 0, 0, mix.size(self.dim) - out.size(self.dim)),
            mode='constant',
            value=0
        )

        return out


# for testing
class TestFuncs(unittest.TestCase):
    def test_encoder(self):
        encoder = Encoder(args)
        en_out = encoder(torch.rand(4, args.enc_in_channels, 512, device=args.device))

        self.assertEqual(en_out.size(), (4, 256, 63))
    
    def test_decoder(self):
        decoder = Decoder(args)
        de_out = decoder(torch.rand(4, 256, 63, device=args.device))

        self.assertEqual(de_out.size(), (4, 1, 512))
    
    def test_ffn(self):
        ffn = FeedForwardNetwork(args)
        ffn_out = ffn(torch.rand(4, 64, args.mha_dim, device=args.device))

        self.assertEqual(ffn_out.size(), (4, 64, 256))
    
    def test_mha(self):
        mha = MultiHeadAttention(args)
        mha_out = mha(torch.rand(4, 64, args.mha_dim, device=args.device))

        self.assertEqual(mha_out.size(), (4, 64, 256))
    
    def test_transformerb(self):
        # input = (batch_size, time, channels)
        transformerb = TransformerBlock(args)
        transformerb_out = transformerb(torch.rand(4, 64, args.mha_dim, device=args.device))

        self.assertEqual(transformerb_out.size(), (4, 64, 256))
    
    def test_spe(self):
        # input = (batch_size, time, channels)
        pe = SinusoidalPositionalEncoding(args)
        pe_out = pe(torch.rand(4, 64, args.mha_dim, device=args.device))

        self.assertEqual(pe_out.size(), (4, 64, 256))
    
    def test_it(self):
        # input = (batch_size, time, channels)
        it = InterTransformer(args)
        it_out = it(torch.rand(4, 64, args.mha_dim, device=args.device))

        self.assertEqual(it_out.size(), (4, 64, 256))
    
    def test_sepformer(self):
        # input = (bach_size, channels, time, chunks)
        sf = SepFormer(args)
        sf_out = sf(torch.rand(4, args.enc_out_channels, 64, 10, device=args.device))

        print("\nSepFormer param size: {}M\n".format(round(sum(p.numel() for p in sf.parameters()) / 1e6)))

        self.assertEqual(sf_out.size(), (4, 256, 64, 10))
    
    def test_masknet(self):
        msknet = MaskNet(args)
        msk_out = msknet(torch.rand(4, args.enc_out_channels, args.chunk_size * 5, device=args.device))

        self.assertEqual(msk_out.size(), (2, 4, 256, 1250))

    def test_inference(self):
        reset_var = args.device
        args.device = torch.device('cpu')

        model = Model(args)

        onnx_export = ONNXInference(model_name='sepformer')

        retVal = onnx_export.convert_torch_to_onnx(
            model=model,
            pass_inputs=True,
            model_inputs=(torch.rand(4, args.enc_in_channels, 2700),),
            input_names=['mix'],
            use_dynamo=True,
            use_external_data=False,
            exist_ok=False
        )

        self.assertEqual(retVal, 0)

        model_inputs = []
        t_sizes = [11232, 4596, 8397, 12371]

        for t in t_sizes:
            model_inputs.append(torch.rand(1, args.enc_in_channels, t))
        
        for model_input in model_inputs:
            model_out = model(model_input)

            onnx_out = onnx_export.inference(
                model_name='sepformer',
                model_input=model_input
            )

            assert_tensors = functools.partial(torch.testing.assert_close, rtol=1.3e-6, atol=1e-5)

            # compare pytorch and onnx output
            assert_tensors(model_out, torch.from_numpy(onnx_out[0]))
    
        args.device = reset_var


def main(out=sys.stderr, verbosity=2): 
    loader = unittest.TestLoader() 
  
    suite = loader.loadTestsFromModule(sys.modules[__name__]) 
    unittest.TextTestRunner(out, verbosity = verbosity).run(suite) 


if __name__ == '__main__':
    args = ModelArgs()

    # model = Model(args)
    # model_out = model(torch.randn(3, args.enc_in_channels, 265, device=args.device))
    # print(model_out.size())

    with open('tests.log', 'w') as f: 
        main(f)
