import math, sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
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

from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder


class DPTransformerModel(TransformerModel):
    def add_args(parser):



class NoOutputBiasTransformerDecoder(FairseqIncrementalDecoder):
    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
         self.adaptive_softmax is None:
            # project back to size of vocabulary
            if not self.share_input_output_embed:
                raise ValueError("Either --share-input-output-embed or --share-all-embeddings is required.")
            return F.linear(features, self.embed_tokens.weight)
        else:
            raise ValueError("--adaptive-softmax can't be set.")



