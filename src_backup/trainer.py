import os
import logging

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, BertConfig

from model import BertWithKnowledgeBase
from schedule import get_schedule
from utils import compute_metrics, get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    Args:
        train_data: TensorDataset
        dev_data: TensorDataset
        test_data: TensorDataset
    """

    def __init__(self, args, train_data=None, dev_data=None, test_data=None):
        self.args = args
        self.device = args.device
        self.writer = SummaryWriter(f'../records/{args.record_path}/log')

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.bert_config = BertConfig.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = BertWithKnowledgeBase(
            self.bert_config, args, self.intent_label_lst, self.slot_label_lst).to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.args.batch_size)

        total_steps = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            }, {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.adam_lr, eps=self.args.adam_epsilon)
        scheduler = get_schedule(args=self.args, optimizer=optimizer, num_training_steps=total_steps)

        # Train!
        logger.info("")
        logger.info("***** Training Start *****")
        logger.info("  Task = %s", f"{self.args.task}")
        logger.info("  Record path = %s", self.args.record_path)
        logger.info("  Random seed = %d", self.args.seed)
        logger.info("  Num epochs = %d", self.args.num_epochs)
        logger.info("  Batch size = %d", self.args.batch_size)
        logger.info("  Total optimization steps = %d", total_steps)
        logger.info("")

        self.model.zero_grad()

        num_epochs = int(self.args.num_epochs)
        for epoch in range(num_epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f'Training {epoch + 1}/{num_epochs}')
            epoch_loss = 0.0
            epoch_steps = 0

            self.model.train()

            for batch in epoch_iterator:
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'intent_label_ids': batch[3],
                    'slot_labels_ids': batch[4],
                    'wn_synset_indexes': batch[5],
                    'wn_synset_lengths': batch[6],
                    'nell_entity_indexes': batch[7],
                    'nell_entity_lengths': batch[8]
                }
                loss, _ = self.model(**inputs)
                epoch_loss += loss.mean().item()
                epoch_steps += 1

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                if epoch_steps % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

                epoch_iterator.set_postfix_str(f'tr_loss={epoch_loss / epoch_steps}')

            self.writer.add_scalar(f'train/loss', epoch_loss / epoch_steps, epoch + 1)

            if (epoch + 1) % 5 == 0:
                # log experiment results.
                self.evaluate('dev', epoch + 1, tensorboard_enabled=True)
            if self.args.save_epochs != 0 and (epoch + 1) % self.args.save_epochs == 0:
                # save model.
                self.save_model(self.args.record_path, epoch + 1)

    def evaluate(self, mode, epoch, tensorboard_enabled):
        if mode == 'test':
            dataset = self.test_data
        elif mode == 'dev':
            dataset = self.dev_data
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        eval_loss = 0.0
        eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'intent_label_ids': batch[3],
                    'slot_labels_ids': batch[4],
                    'wn_synset_indexes': batch[5],
                    'wn_synset_lengths': batch[6],
                    'nell_entity_indexes': batch[7],
                    'nell_entity_lengths': batch[8]
                }
                loss, (intent_logits, slot_logits) = self.model(**inputs)
                eval_loss += loss.mean().item()

            eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with aits_best index directly
                    crf_results = self.model.crf.decode(slot_logits.transpose(0, 1),
                                                        mask=inputs['attention_mask'].byte().transpose(0, 1))
                    slot_preds = np.array(np.array(crf_results))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs['slot_labels_ids'].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    crf_results = self.model.crf.decode(slot_logits.transpose(0, 1),
                                                        mask=inputs['attention_mask'].byte().transpose(0, 1))
                    slot_preds = np.append(slot_preds, np.array(crf_results), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(
                    out_slot_labels_ids, inputs['slot_labels_ids'].detach().cpu().numpy(), axis=0)

        # Intent result.
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result.
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        results = {
            "loss": eval_loss / eval_steps,
            "epoch": epoch
        }
        results.update(
            compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list))
        logger.info(f"Eval results - epoch={epoch},\t{mode},\t"
                    f"intent_acc={'%.5f' % results['intent_acc']},\t"
                    f"slot_f1={'%.5f' % results['slot_f1']},\t"
                    f"semantic_acc={'%.5f' % results['semantic_acc']},\t"
                    f"loss={'%.5f' % results['loss']}")
        logger.info("")

        if tensorboard_enabled:
            self.writer.add_scalar(f'{mode}/intent_acc', results["intent_acc"], epoch)
            self.writer.add_scalar(f'{mode}/slot_f1', results["slot_f1"], epoch)
            self.writer.add_scalar(f'{mode}/semantic_acc', results["semantic_acc"], epoch)
            self.writer.add_scalar(f'{mode}/loss', results["loss"], epoch)

    def predict(self, texts, tokenizer):
        batch = self._convert_texts_to_tensors(texts, tokenizer)

        slot_label_mask = batch[3]
        self.model.eval()

        # We have only one batch
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                intent_label_ids=None,
                slot_labels_ids=None,
                wn_synset_indexes=batch[5],
                wn_synset_length=batch[6],
                nell_entity_indexes=batch[7],
                nell_entity_lengths=batch[8]
            )
            _, (intent_logits, slot_logits) = outputs[:2]  # loss doesn't needed

        # Intent prediction
        intent_preds = intent_logits.detach().cpu().numpy()
        intent_preds = np.argmax(intent_preds, axis=1)
        intent_list = []
        for intent_index in intent_preds:
            intent_list.append(self.intent_label_lst[intent_index])

        # Slot prediction
        if self.args.use_crf:
            slot_preds = np.array(self.model.crf.decode(slot_logits))
        else:
            slot_preds = slot_logits.detach().cpu().numpy()
            slot_preds = np.argmax(slot_preds, axis=2)

        out_slot_labels_ids = slot_label_mask.detach().cpu().numpy()

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        # Make output.txt with texts, intent_list and slot_preds_list
        with open(os.path.join(self.args.pred_dir, self.args.pred_output_file), 'w', encoding='utf-8') as f:
            for text, intent, slots in zip(texts, intent_list, slot_preds_list):
                f.write("text: {}\n".format(text))
                f.write("intent: {}\n".format(intent))
                f.write("slots: {}\n".format(' '.join(slots)))
                f.write("\n")

        # print output.json
        with open(os.path.join(self.args.pred_dir, self.args.pred_output_file), 'r', encoding='utf-8') as f:
            print(f.read())

    def _convert_texts_to_tensors(self, texts, tokenizer,
                                  cls_token_segment_id=0,
                                  pad_token_segment_id=0,
                                  sequence_a_segment_id=0,
                                  mask_padding_with_zero=True):
        """
        Only add input_ids, attention_mask, token_type_ids
        Labels aren't required.
        """
        # Setting based on the current models type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
        slot_label_mask_batch = []

        for text in texts:
            tokens = []
            slot_label_mask = []
            for word in text.split():
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [unk_token]  # For handling the bad-encoded word
                tokens.extend(word_tokens)
                # Real label position as 0 for the first token of the word, and padding ids for the remaining tokens
                slot_label_mask.extend([0] + [self.pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > self.args.max_seq_len - special_tokens_count:
                tokens = tokens[:(self.args.max_seq_len - special_tokens_count)]
                slot_label_mask = slot_label_mask[:(self.args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            slot_label_mask += [self.pad_token_label_id]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            slot_label_mask = [self.pad_token_label_id] + slot_label_mask
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            slot_label_mask = slot_label_mask + ([self.pad_token_label_id] * padding_length)

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            token_type_ids_batch.append(token_type_ids)
            slot_label_mask_batch.append(slot_label_mask)

        # Making tensor that is batch size of 1
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long).to(self.device)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long).to(self.device)
        token_type_ids_batch = torch.tensor(token_type_ids_batch, dtype=torch.long).to(self.device)
        slot_label_mask_batch = torch.tensor(slot_label_mask_batch, dtype=torch.long).to(self.device)

        return input_ids_batch, attention_mask_batch, token_type_ids_batch, slot_label_mask_batch

    def save_model(self, record_path, epoch):
        # Save models checkpoints (Overwrite)
        output_dir = os.path.join("..", "records", record_path, "models", str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("")

    def load_model(self, path):
        # Check whether models exists
        model_dir = os.path.join("..", "records", path)

        if not os.path.exists(model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = BertConfig.from_pretrained(model_dir)
            self.model = BertWithKnowledgeBase.from_pretrained(
                model_dir,
                config=self.bert_config,
                args=self.args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst
            ).to(self.device)
        except:
            raise Exception("Some models files might be missing...")
