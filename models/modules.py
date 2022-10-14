from misc.utils import ddict

from transformers.modeling_utils import PreTrainedModel
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel
)

from models.tabformer_tokenizer import TabFormerTokenizer
from models.hierarchical import TabFormerEmbeddings
from models.tabformer_bert import TabFormerBertForMaskedLM, TabFormerBertConfig
from models.tabformer_gpt2 import TabFormerGPT2LMHeadModel


class TabFormerBaseModel(PreTrainedModel):
    def __init__(self, hf_model, tab_embeddings, config):
        super().__init__(config)

        self.model = hf_model
        self.tab_embeddings = tab_embeddings

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerHierarchicalLM(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)
    
class TabFormerHierarchicalClassification(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = BertForSequenceClassification(self.config)

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerBertLM:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, 
                 field_hidden_size=64, tab_embeddings_num_attention_heads=8, num_attention_heads=12,
                 hidden_size=768, tab_embedding_num_encoder_layers=1, tab_embedding_dropout=0.1):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        
        assert field_hidden_size % tab_embeddings_num_attention_heads == 0, \
               "\"field_hidden_size\" must be divisible by \"tab_embeddings_num_attention_heads\""
            
        assert hidden_size % num_attention_heads == 0, \
               "\"hidden_size\" must be divisible by \"num_attention_heads\""
        
        assert hidden_size % ncols == 0, \
               "\"hidden_size\" must be divisible by \"ncols\""

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          tab_embeddings_num_attention_heads=tab_embeddings_num_attention_heads,
                                          tab_embedding_num_encoder_layers=tab_embedding_num_encoder_layers,
                                          tab_embedding_dropout=tab_embedding_dropout,
                                          num_attention_heads=num_attention_heads,
                                          flatten=flatten)
        
        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):

        if flatten and not field_ce:
            # flattened vanilla BERT
            model = BertForMaskedLM(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            model = TabFormerBertForMaskedLM(self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            model = TabFormerHierarchicalLM(self.config, self.vocab)

        return model
    
class TabFormerBertClassification(TabFormerBertLM):
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, 
                 field_hidden_size=64, tab_embeddings_num_attention_heads=8, num_attention_heads=12,
                 hidden_size=768, tab_embedding_num_encoder_layers=1, tab_embedding_dropout=0.1, num_labels=None):

        self.num_labels = num_labels
        
        super().__init__(special_tokens, vocab, field_ce, flatten, ncols, 
                         field_hidden_size, tab_embeddings_num_attention_heads, num_attention_heads,
                         hidden_size, tab_embedding_num_encoder_layers, tab_embedding_dropout)

    def get_model(self, field_ce, flatten):

        self.config.num_labels = self.num_labels
        
        if flatten and not field_ce:
            # flattened vanilla BERT
            model = BertForSequenceClassification(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            model = BertForSequenceClassification(self.config)
        else:
            # hierarchical field CE BERT
            model = TabFormerHierarchicalClassification(self.config, self.vocab)

        return model



class TabFormerGPT2:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False):

        self.vocab = vocab
        self.config = GPT2Config(vocab_size=len(self.vocab))

        self.tokenizer = TabFormerTokenizer(
            unk_token=special_tokens.unk_token,
            bos_token=special_tokens.bos_token,
            eos_token=special_tokens.eos_token
        )

        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):
        if field_ce:
            model = TabFormerGPT2LMHeadModel(self.config, self.vocab)
        else:
            model = GPT2LMHeadModel(self.config)
        if not flatten:
            tab_emb_config = ddict(vocab_size=len(self.vocab), hidden_size=self.config.hidden_size)
            model = TabFormerBaseModel(model, TabFormerEmbeddings(tab_emb_config))

        return model
