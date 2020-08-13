import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel

from crf import CRF
from layers import KnowledgeIntegrator, KB_EMBEDDING_DIM, Classifier, BertEncoder, KnowledgeDecoder


class BertWithKnowledgeBase(BertPreTrainedModel):

    def __init__(self, bert_config, args, intent_label_lst, slot_label_lst):
        super(BertWithKnowledgeBase, self).__init__(bert_config)

        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.slot_pad_token_index = slot_label_lst.index(args.pad_label)

        self.encoder = BertEncoder(args, bert_config=bert_config)

        self.knowledge_integrator = KnowledgeIntegrator(args,
                                                        input_dim=bert_config.hidden_size,
                                                        pos_embed_enabled=args.pos)
        self.knowledge_decoder = KnowledgeDecoder(args, input_dim=bert_config.hidden_size)

        self.intent_classifier = Classifier(args,
                                            input_size=bert_config.hidden_size + 2 * KB_EMBEDDING_DIM,
                                            num_labels=self.num_intent_labels)
        self.slot_classifier = Classifier(args,
                                          input_size=bert_config.hidden_size + 2 * KB_EMBEDDING_DIM,
                                          num_labels=self.num_slot_labels)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                intent_label_ids,
                slot_labels_ids,
                wn_synset_indexes,
                wn_synset_lengths,
                nell_entity_indexes,
                nell_entity_lengths):
        """
        Args:
            input_ids: Tensor[batch_size, seq_len]
            attention_mask: Tensor[batch_size, max_seq_len]
            token_type_ids: Tensor[batch_size, max_seq_len]
            intent_label_ids: Tensor[batch_size]
            slot_labels_ids: Tensor[batch_size, max_seq_len]
            wn_synset_indexes: Index of WordNet synsets. Tensor[batch_size, max_seq_len, concept_count]
            wn_synset_lengths: Real lengths of WordNet synsets for each token. Tensor[batch_size, max_seq_len]
            nell_entity_indexes: Index of NELL entities. Tensor[batch_size, max_seq_len, concept_count]
            nell_entity_lengths: Real lengths of NELL entities for each token. Tensor[batch_size, max_seq_len]

        Return:
            total_loss, (
                intent_logits: Tensor[batch, num_inte]),
                slot_logits: Tensor[batch, seq_len, num_slot])
            )
        """

        """ 1. Bert & Intent Decoding (uni-LSTM). """
        # [batch_size, max_seq_len, hidden_size]
        intent_features, slot_features = self.encoder(input_ids=input_ids,
                                                      attention_mask=attention_mask,
                                                      token_type_ids=token_type_ids)

        """ 2. Knowledge integration. """
        # [batch_size, max_seq_len, KB_EMBEDDING_DIM]
        knowledge, knowledge_contexts = self.knowledge_integrator(intent_features=intent_features,
                                                                  slot_features=slot_features,
                                                                  attention_mask=attention_mask,
                                                                  wn_synset_indexes=wn_synset_indexes,
                                                                  wn_synset_lengths=wn_synset_lengths,
                                                                  nell_entity_indexes=nell_entity_indexes,
                                                                  nell_entity_lengths=nell_entity_lengths)

        """ 3. Knowledge decoding. """
        # [batch_size, max_seq_len, hidden_size + 2 * KB_EMBEDDING_DIM]
        intent_y, slot_y = self.knowledge_decoder(intent_features=intent_features,
                                                  slot_features=slot_features,
                                                  knowledge=knowledge,
                                                  knowledge_contexts=knowledge_contexts,
                                                  attention_mask=attention_mask)

        """ 4. Classification. """
        intent_logits = self.intent_classifier(intent_y)  # [batch_size, max_seq_len, num_intent_labels]
        slot_logits = self.slot_classifier(slot_y)  # [batch_size, max_seq_len, num_slot_labels]

        """ 5. Loss. """
        total_loss = 0
        active_tokens = slot_labels_ids != self.args.ignore_index

        # Intent Softmax.
        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)

            active_loss = active_tokens.view(-1)
            active_logits = intent_logits.view(-1, self.num_intent_labels)[active_loss]
            active_labels = intent_label_ids.unsqueeze(1).expand(-1, input_ids.shape[1]).reshape(-1)[active_loss]

            total_loss += intent_loss_fct(active_logits, active_labels)

        # Slot Softmax.
        if slot_labels_ids is not None:
            if self.args.use_crf:
                # Make new slot_labels_ids, changing ignore_index(-100) to PAD index in slot label
                # In torch-crf, if index is lower than 0, it makes error when indexing the list
                padded_slot_labels_ids = slot_labels_ids.detach().clone()
                padded_slot_labels_ids[padded_slot_labels_ids == self.args.ignore_index] = self.slot_pad_token_index

                slot_loss = -1 * self.crf(
                    slot_logits.transpose(0, 1),
                    padded_slot_labels_ids.transpose(0, 1),
                    mask=attention_mask.byte().transpose(0, 1),
                    reduction='mean'
                )  # negative log-likelihood.
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)

                active_loss = active_tokens.view(-1)
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]

                slot_loss = slot_loss_fct(active_logits, active_labels)

            total_loss += self.args.slot_loss_coef * slot_loss

        # intent_logits = F.softmax(intent_logits, dim=2)  # [batch_size, max_seq_len, num_intent_labels]
        # intent_logits = intent_logits.masked_fill(
        #     active_tokens.unsqueeze(2).expand(-1, -1, intent_logits.shape[2]) == False, 0)  # [batch, seq_len, num_inte]
        # intent_logits = torch.sum(intent_logits, dim=1, keepdim=False)  # [batch_size, num_intent_labels]

        # intent logits.
        intent_logits = F.softmax(intent_logits, dim=2)  # [batch_size, max_seq_len, num_intent_labels]
        intent_argmax = torch.argmax(intent_logits, dim=2, keepdim=True)  # [batch_size, max_seq_len, 1]
        # [batch_size, max_seq_len, num_intent_labels]
        intent_logits_1 = torch.zeros(intent_logits.shape).to(self.args.device).scatter_(
            dim=-1,
            index=intent_argmax,
            src=torch.ones(intent_argmax.shape).to(self.args.device)
        )
        intent_logits_1 = intent_logits_1.masked_fill(
            active_tokens.unsqueeze(2).expand(-1, -1, intent_logits_1.shape[2]) == False, 0)  # [batch, seq_len, num_inte]
        intent_logits_1 = torch.sum(intent_logits_1, dim=1, keepdim=False)  # [batch_size, num_intent_labels]

        intent_logits_2 = intent_logits.masked_fill(
            active_tokens.unsqueeze(2).expand(-1, -1, intent_logits.shape[2]) == False, 0)  # [batch, seq_len, num_inte]
        intent_logits_2 = torch.sum(intent_logits_2, dim=1, keepdim=False)  # [batch_size, num_intent_labels]
        intent_logits_2 /= 1.0 * self.args.max_seq_len

        intent_logits = intent_logits_1 + intent_logits_2

        # slot logits.
        slot_logits = F.softmax(slot_logits, dim=2)  # [batch_size, max_seq_len, num_slot_labels]

        return total_loss, (intent_logits, slot_logits)
