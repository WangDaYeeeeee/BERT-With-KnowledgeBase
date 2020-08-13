import torch
import torch.nn as nn

import numpy as np
from transformers import BertModel

from attention import Attention
from coder import LSTMDecoder
from utils import get_wn_concept_vectors, get_nell_concept_vectors

KB_EMBEDDING_DIM = 100

PAD_WN_CONCEPT = '__PAD'
PAD_WN_CONCEPT_ID = 0

PAD_NELL_CONCEPT = 'concept:pad'
PAD_NELL_CONCEPT_ID = 0


class BertEncoder(nn.Module):

    def __init__(self, args, bert_config):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=bert_config)
        if args.uni_intent is False:
            self.intent_decoder = LSTMDecoder(args=args,
                                              input_size=bert_config.hidden_size,
                                              hidden_size=bert_config.hidden_size,
                                              dropout_rate=args.intent_decoder_dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Args:
            input_ids: Tensor[batch_size, seq_len]
            attention_mask: Tensor[batch_size, max_seq_len]
            token_type_ids: Tensor[batch_size, max_seq_len]

        Return:
            intent_features: Tensor[batch_size, max_seq_len, hidden_size]
            slot_features: Tensor[batch_size, max_seq_len, hidden_size]
        """

        # slot_features, intent_features, (hidden_states), (attentions)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # [batch_size, max_seq_len, hidden_size]
        slot_features = bert_outputs[0]

        if self.args.uni_intent:
            # [batch_size, max_seq_len, hidden_size]
            intent_features = bert_outputs[1].unsqueeze(1).repeat(1, slot_features.shape[1], 1)
        else:
            # [CLS], [1, batch_size, hidden_size]
            intent_features = bert_outputs[1].unsqueeze(0)
            # [batch_size, max_seq_len - 2, hidden_size]
            intent_features = self.intent_decoder(seq_input=slot_features[:, 1:-1, :],
                                                  seq_lens=torch.sum(attention_mask, dim=1, keepdim=False) - 2,
                                                  h_input=intent_features,
                                                  c_input=intent_features)
            # [batch_size, max_seq_len, hidden_size]
            intent_features = torch.cat([
                torch.zeros(intent_features.shape[0], 1, intent_features.shape[2]).to(self.args.device),
                intent_features,
                torch.zeros(intent_features.shape[0], 1, intent_features.shape[2]).to(self.args.device)
            ], dim=1)

        return intent_features, slot_features


class KnowledgeIntegrator(nn.Module):
    """
    Args:
        input_dim: Last dimension of input tensor.
        pos_embed_enabled: Whether to enable positional embedding.
    """

    def __init__(self, args, input_dim, pos_embed_enabled=True):
        super(KnowledgeIntegrator, self).__init__()
        self.max_concepts_count = args.max_wn_concepts_count + args.max_nell_concepts_count
        self.top_concepts_count = args.top_concepts_count

        if self.max_concepts_count > 0:
            wn_concept_vectors = np.array(get_wn_concept_vectors(args))
            self.wn_concept_embed = nn.Embedding(num_embeddings=len(wn_concept_vectors),
                                                 embedding_dim=KB_EMBEDDING_DIM,
                                                 padding_idx=PAD_WN_CONCEPT_ID)
            self.wn_concept_embed.weight = nn.Parameter(torch.from_numpy(wn_concept_vectors), requires_grad=False)

            nell_concept_vectors = np.array(get_nell_concept_vectors(args))
            self.nell_concept_embed = nn.Embedding(num_embeddings=len(nell_concept_vectors),
                                                   embedding_dim=KB_EMBEDDING_DIM,
                                                   padding_idx=PAD_NELL_CONCEPT_ID)
            self.nell_concept_embed.weight = nn.Parameter(torch.from_numpy(nell_concept_vectors), requires_grad=False)

            self.knowledge_attn = Attention(query_dim=2 * input_dim,
                                            key_dim=KB_EMBEDDING_DIM,
                                            method=args.attn,
                                            dropout_rate=args.knowledge_attn_dropout)
            self.pos_embed = PositionalEmbedding(embed_dim=KB_EMBEDDING_DIM,
                                                 max_seq_len=args.max_seq_len) if pos_embed_enabled else None

            self.context_attn = Attention(query_dim=2 * input_dim,
                                          key_dim=KB_EMBEDDING_DIM,
                                          method=args.attn,
                                          dropout_rate=args.context_attn_dropout)
            self.layer_norm = nn.LayerNorm(KB_EMBEDDING_DIM)

    def forward(self, intent_features, slot_features, attention_mask,
                wn_synset_indexes, wn_synset_lengths, nell_entity_indexes, nell_entity_lengths):
        """
        Args:
            intent_features: Tensor[batch_size, seq_len, input_dim]
            slot_features: Tensor[batch_size, seq_len, input_dim]
            attention_mask: Tensor[batch_size, max_seq_len]
            wn_synset_indexes: Tensor[batch_size, max_seq_len, concept_count]
            wn_synset_lengths: Tensor[batch_size, max_seq_len]
            nell_entity_indexes: Tensor[batch_size, max_seq_len, concept_count]
            nell_entity_lengths: Tensor[batch_size, max_seq_len]

        Return:
            knowledge, contexts
            output state which contains knowledge and knowledge contexts. 2 * Tensor[batch, seq_len, KB_EMBEDDING_DIM]
        """

        if self.max_concepts_count <= 0:
            # return zero tensor if no any knowledge enabled.
            zeros = torch.zeros(attention_mask.shape[0], attention_mask.shape[1], KB_EMBEDDING_DIM).to(
                self.get_device(attention_mask))
            return zeros, zeros

        # get concept embedding vectors. [batch_size, seq_len, concept_count, KB_EMBEDDING_DIM]
        concept_vectors = torch.cat([
            self.wn_concept_embed(wn_synset_indexes),
            self.nell_concept_embed(nell_entity_indexes)
        ], dim=2)

        # [batch_size, seq_len, 2 * input_dim]
        token_features = torch.cat([slot_features, intent_features], dim=2)

        knowledge = []  # [seq_len, Tensor[batch_size, KB_EMBEDDING_DIM]]
        for i in range(token_features.shape[1]):  # 1, 2, ..., seq_len - 1
            query = token_features[:, i, :].unsqueeze(1)  # [batch_size, 1, 2 * input_dim]
            keys = concept_vectors[:, i, :, :]  # [batch_size, concept_count, KB_EMBEDDING_DIM]
            mask = torch.cat([
                wn_synset_indexes[:, i, :],
                nell_entity_indexes[:, i, :]
            ], dim=1).unsqueeze(1)  # [batch_size, 1, concept_count]

            # [batch_size, KB_EMBEDDING_DIM]
            knowledge.append(
                self.knowledge_attn(
                    queries=query,
                    keys=keys.float(),
                    mask=mask,
                    top_k=self.top_concepts_count if self.top_concepts_count < self.max_concepts_count else None
                ).squeeze(1)
            )

        # [batch_size, seq_len, KB_EMBEDDING_DIM]
        knowledge = torch.stack(knowledge, dim=1)
        # [batch_size, max_seq_len]
        knowledge_mask = wn_synset_lengths + nell_entity_lengths
        # knowledge_mask = attention_mask

        if self.pos_embed is not None:
            knowledge += self.pos_embed(knowledge_mask)

        # [batch_size, max_seq_len, KB_EMBEDDING_DIM]
        contexts = self.context_attn(queries=token_features,
                                     keys=knowledge,
                                     mask=knowledge_mask.unsqueeze(1).expand(-1, knowledge_mask.shape[1], -1))
        contexts = self.layer_norm(contexts)

        return knowledge, contexts

    @staticmethod
    def get_device(t):
        try:
            device_id = t.get_device()
        except:
            return 'cpu'
        else:
            return 'cpu' if device_id < 0 else 'cuda'


class PositionalEmbedding(nn.Module):

    def __init__(self, embed_dim, max_seq_len):
        super(PositionalEmbedding, self).__init__()

        position_encodings = np.array(
            [[pos / np.power(10000, 2. * (i // 2) / embed_dim) for i in range(embed_dim)] for pos in
             range(max_seq_len)])
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])

        # [max_seq_len + 1, embed_dim]
        self.position_embed = nn.Parameter(torch.tensor(position_encodings), requires_grad=False)

    def forward(self, mask):
        """
        Args:
            mask: Use none zero as valid value flag and 0 as pad flag. Tensor[batch_size, max_seq_len]

        Return:
            Tensor[batch, max_seq_len, embed_dim]
        """

        # [batch_size, max_seq_len, embed_dim]
        mask = mask.unsqueeze(-1).expand(-1, -1, self.position_embed.shape[-1])
        # [batch_size, max_seq_len, embed_dim]
        embeddings = self.position_embed.unsqueeze(0).expand(mask.shape[0], -1, -1)
        return embeddings.masked_fill(mask == 0, 0).float()


class KnowledgeDecoder(nn.Module):
    """
    Args:
        input_dim: Last dimension of input tensor.
    """

    def __init__(self, args, input_dim):
        super(KnowledgeDecoder, self).__init__()
        self.args = args
        self.intent_decoder = LSTMDecoder(args,
                                          input_size=2 * input_dim + KB_EMBEDDING_DIM,
                                          hidden_size=KB_EMBEDDING_DIM,
                                          bidirectional=True,
                                          dropout_rate=args.knowledge_decoder_dropout)
        self.slot_decoder = LSTMDecoder(args,
                                        input_size=2 * input_dim + KB_EMBEDDING_DIM * (1 if args.unstack else 3),
                                        hidden_size=KB_EMBEDDING_DIM,
                                        bidirectional=True,
                                        dropout_rate=args.knowledge_decoder_dropout)

        self.layer_norm = nn.LayerNorm(2 * KB_EMBEDDING_DIM)

    def forward(self, intent_features, slot_features, knowledge, knowledge_contexts, attention_mask):
        """
        Args:
            intent_features: Tensor[batch_size, seq_len, input_dim]
            slot_features: Tensor[batch_size, seq_len, input_dim]
            knowledge: Tensor[batch_size, seq_len, KB_EMBEDDING_DIM]
            knowledge_contexts: Tensor[batch_size, seq_len, KB_EMBEDDING_DIM]
            attention_mask: Tensor[batch_size, max_seq_len]

        Return:
            intent_y: Tensor[batch_size, max_seq_len, input_dim + 2 * KB_EMBEDDING_DIM]
            slot_y: Tensor[batch_size, max_seq_len, input_dim + 2 * KB_EMBEDDING_DIM]
        """

        seq_lens = torch.sum(attention_mask, dim=1, keepdim=False) - 2
        padding = torch.zeros(intent_features.shape[0], 1, 2 * KB_EMBEDDING_DIM).to(self.args.device)

        # [batch_size, max_seq_len - 2, 2 * KB_EMBEDDING_DIM]
        intent_hidden = self.intent_decoder(
            seq_input=torch.cat([intent_features, slot_features, knowledge_contexts], dim=2)[:, 1:-1, :],
            seq_lens=seq_lens
        )
        intent_hidden = self.layer_norm(intent_hidden)
        # [batch_size, max_seq_len, 2 * KB_EMBEDDING_DIM]
        intent_hidden = torch.cat([padding, intent_hidden, padding], dim=1)

        if self.args.unstack:
            # [batch_size, max_seq_len - 2, 2 * input_dim + KB_EMBEDDING_DIM]
            slot_input = torch.cat([intent_features, slot_features, knowledge_contexts], dim=2)[:, 1:-1, :]
        else:
            # [batch_size, max_seq_len - 2, 2 * input_dim + 3 * KB_EMBEDDING_DIM]
            slot_input = torch.cat([intent_features, slot_features, intent_hidden, knowledge_contexts],
                                   dim=2)[:, 1:-1, :]
        # [batch_size, max_seq_len - 2, 2 * KB_EMBEDDING_DIM]
        slot_hidden = self.slot_decoder(seq_input=slot_input, seq_lens=seq_lens)

        slot_hidden = self.layer_norm(slot_hidden)
        # [batch_size, max_seq_len, 2 * KB_EMBEDDING_DIM]
        slot_hidden = torch.cat([padding, slot_hidden, padding], dim=1)

        # [batch_size, max_seq_len, input_dim + 2 * KB_EMBEDDING_DIM]
        intent_y = torch.cat([intent_features, intent_hidden], dim=2)
        slot_y = torch.cat([slot_features, slot_hidden], dim=2)

        return intent_y, slot_y


class Classifier(nn.Module):

    def __init__(self, args, input_size, num_labels):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(args.classifier_dropout)
        self.linear = nn.Linear(input_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
