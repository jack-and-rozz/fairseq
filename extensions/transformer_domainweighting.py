#

from .transformer_finetuning import base_architecture, TransformerModelForFinetuning

from fairseq.models import (
    register_model,
    register_model_architecture,
)


@register_model('transformer_domainweighting')
class DomainAwareTransformerModel(TransformerModelForFinetuning):
    pass


@register_model_architecture('transformer_domainweighting', 
                             'transformer_domainweighting')
def transformer_domainweighting(args):
    base_architecture(args)
