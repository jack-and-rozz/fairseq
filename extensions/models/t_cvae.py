# 
import math, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm, MultiheadAttention

from fairseq import options, utils

from fairseq.models.transformer import (
    TransformerModel, 
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    Embedding, 
    Linear,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture as _base_architecture,
) 

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


@register_model('t_cvae')
class TransformerCVAE(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # * modified *
        parser.add_argument('--disable-sharing-decoder', action='store_true',
                            help='if set, the encoder and decoder have independent parameters each other.')

        # fmt: on

    def __init__(self, encoder, decoder, latent, 
                 left_pad_source, left_pad_target,
                 extra_feature_dicts={}, turn_delimiter='▁<eot>'):
        super().__init__(encoder, decoder)
        self.extra_feature_dicts = extra_feature_dicts
        self.latent = latent
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.turn_delimiter = turn_delimiter

    @classmethod
    def build_model(cls, args, task):

        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)

            # if provided, load from preloaded dictionaries
            if path:
                # embed_dict = utils.parse_embedding(path)
                embed_dict = parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)

            # if args.disable_training_embeddings:
            #     emb.weight.requires_grad = False
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path,
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if not args.disable_sharing_decoder:
            assert args.share_all_embeddings
            assert args.encoder_attention_heads == args.decoder_attention_heads
            assert args.encoder_embed_dim == args.decoder_embed_dim
            assert args.encoder_layers == args.decoder_layers
            assert args.encoder_layers == args.decoder_layers
            assert args.encoder_ffn_embed_dim == args.decoder_ffn_embed_dim

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        latent = CVAELatent(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, 
                                    encoder)
        return cls(encoder, decoder, latent, 
                   args.left_pad_source, args.left_pad_target,
                   extra_feature_dicts=task.extra_feature_dicts)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, encoder):
        # if args.disable_sharing_decoder:
        if args.disable_sharing_decoder or True: # DEBUG
            return TransformerDecoder(args, tgt_dict, embed_tokens)
        else:
            return SharedTransformerDecoder(args, tgt_dict, embed_tokens, 
                                            encoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """


        # Create joined sequences of ids (src + <eot> + tgt) to compute posterior distributions.
        # TODO: concat src and tgt -> encode or separately encode and concat repls?
        device = src_tokens.device
        batch_size = src_tokens.shape[0]
        encoder_pad_idx = self.encoder.dictionary.pad()
        decoder_pad_idx = self.decoder.dictionary.pad()

        eot_idx = self.decoder.dictionary.indices[self.turn_delimiter]

        tgt_lengths = prev_output_tokens.shape[1] - ((prev_output_tokens - decoder_pad_idx) == 0).sum(dim=1)
        joined_lengths = src_lengths + tgt_lengths + 1
        max_length = torch.max(joined_lengths)
        joined_tokens = torch.ones([batch_size, max_length], dtype=torch.int64, device=device) * encoder_pad_idx

        for i in range(batch_size):
            ls = src_lengths[i]
            lt = tgt_lengths[i]
            tgt_tok_ids = prev_output_tokens[i, prev_output_tokens.shape[1]-lt:] if self.left_pad_target else prev_output_tokens[i, :lt]

            if self.left_pad_source:
                src_tok_ids = src_tokens[i, src_tokens.shape[1]-ls:]
                offset = max_length - joined_lengths[i]
                joined_tokens[i, offset:offset+src_lengths[i]] = src_tok_ids
                joined_tokens[i, offset+src_lengths[i]] = eot_idx 
                joined_tokens[i, offset+src_lengths[i]+1:] = tgt_tok_ids
            else:
                src_tok_ids = src_tokens[i, :ls]
                joined_tokens[i, :src_lengths[i]] = src_tok_ids
                joined_tokens[i, src_lengths[i]] = eot_idx 
                joined_tokens[i, src_lengths[i]:src_lengths[i]+tgt_lengths[i]] = tgt_tok_ids

        # DEBUG
        # n = 1
        # print(src_tokens[n])
        # print(prev_output_tokens[n])
        # print(joined_tokens[n])
        # print(self.encoder.dictionary.string(src_tokens[n]))
        # print(self.decoder.dictionary.string(prev_output_tokens[n]))
        # print(self.encoder.dictionary.string(joined_tokens[n]))
        # print(src_tokens.device, src_lengths.device)
        # print(joined_tokens.device, joined_lengths.device)
        # exit(1)

        # encoder_out: {'encoder_out': [seq_len, batch, hidden], 'encoder_padding_mask': [seq_len, batch, hidden]}
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        joined_encoder_out = self.encoder(joined_tokens, src_lengths=joined_lengths, **kwargs)
        latent_out = self.latent(encoder_out, joined_encoder_out)

        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        print(latent_out)
        # print(decoder_out)
        exit(1)

        return decoder_out

class CVAELatent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.latent_dim = latent_dim = args.decoder_embed_dim
        self.prior_net = nn.Linear(args.encoder_embed_dim, latent_dim * 2, bias=False)
        self.post_net = nn.Linear(args.encoder_embed_dim, latent_dim * 2, bias=False)
        self.attn = MultiheadAttention(
            args.encoder_embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=False)
        self.query = nn.Parameter(torch.Tensor(1, 1, args.encoder_embed_dim)) 

    def forward(self, prior_out, post_out):
        batch_size = prior_out['encoder_out'].shape[1]

        query = self.query.repeat(1, batch_size, 1)
        prior_attn_out, _ = self.attn(
            query=query,
            key=prior_out['encoder_out'],
            value=prior_out['encoder_out'],
            key_padding_mask=prior_out['encoder_padding_mask'])
        prior_mean, prior_logvar = torch.split(self.prior_net(prior_attn_out[0]),
                                               self.latent_dim, dim=-1)

        if self.training:
            post_attn_out, _ = self.attn(
                query=query,
                key=post_out['encoder_out'],
                value=post_out['encoder_out'],
                key_padding_mask=post_out['encoder_padding_mask'])
            post_mean, post_logvar = torch.split(self.post_net(post_attn_out[0]),
                                                 self.latent_dim, dim=-1)
            z_mean = post_mean
            z_logvar = post_logvar
            # TODO: slow?
            # q = torch.distributions.Normal(post_mean, post_logvar)
            # p = torch.distributions.Normal(prior_mean, prior_logvar)
            # kl_loss = torch.distributions.kl_divergence(p, q)
            kld = self.gaussian_kld(post_mean, post_logvar, 
                                    prior_mean, prior_logvar)
        else:
            z_mean = prior_mean
            z_logvar = prior_logvar
            kld = None
        z = torch.randn(z_mean.size(), device=z_mean.device) * torch.exp(0.5 * z_logvar) + z_mean
        return z, kld

    @staticmethod
    def gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (post_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - post_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(post_logvar), torch.exp(prior_logvar)), dim=-1)
        return kld


@register_model_architecture('t_cvae', 't_cvae')
def base_architecture(args):
    _base_architecture(args)
    args.disable_sharing_decoder = getattr(args, 'disable_sharing_decoder', False)



class SharedTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, encoder_layer, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )
        # self.self_attn = encoder_layer.self_attn
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, 'encoder_embed_dim', None),
                vdim=getattr(args, 'encoder_embed_dim', None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

class SharedTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, encoder,
                 no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, encoder.layers[i], no_encoder_attn)
            for i in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        




def parse_embedding(embed_path):
    embed_dict = {}
    with open(embed_path) as f_embed:
        line = f_embed.readline()
        while line:
            pieces = line.strip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor([float(weight) for weight in pieces[1:]])
            try:
                line = f_embed.readline()
            except Exception as e:
                print(e, file=sys.stderr)
                continue

    return embed_dict
