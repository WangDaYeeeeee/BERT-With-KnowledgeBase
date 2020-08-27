import logging
import os

import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm

from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class Predictor(object):
    """
    Args:
        model: Module
        test_data: TensorDataset
    """

    def __init__(self, args, model, test_data):
        self.args = args
        self.device = args.device

        self.model = model
        self.test_text = [line.strip() for line in open(os.path.join(
            self.args.data_dir, 'dataset', self.args.task, 'test', 'seq.in'), "r", encoding="utf-8")]
        self.test_data = test_data
        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

    def predict(self):
        eval_sampler = SequentialSampler(self.test_data)
        eval_dataloader = DataLoader(self.test_data, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Pred!
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Predicting"):
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
                _, (intent_logits, slot_logits) = self.model(**inputs)

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
                slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_labels_ids = inputs['slot_labels_ids'].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                out_slot_labels_ids = np.append(
                    out_slot_labels_ids, inputs['slot_labels_ids'].detach().cpu().numpy(), axis=0)

        # Intent result.
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result.
        slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        f_out = os.path.join(self.args.pred_dir, f'{self.args.task}_{self.args.pred_output_file}')
        with open(f_out, 'w', encoding='utf-8') as f_out:
            for sentence, intent_pred_id, intent_out_id, slot_pred_id_list, slot_out_id_list in zip(
                    self.test_text, intent_preds, out_intent_label_ids, slot_preds, out_slot_labels_ids):

                f_out.write(f'{sentence}\n')

                slot_pred_id_list = slot_pred_id_list.tolist()
                slot_out_id_list = slot_out_id_list.tolist()
                different = False
                for i in range(len(slot_out_id_list) - 1, -1, -1):  # from rear to head, remove all ignored items.
                    if slot_out_id_list[i] == self.args.ignore_index:
                        slot_pred_id_list.pop(i)
                        slot_out_id_list.pop(i)
                    elif slot_out_id_list[i] != slot_pred_id_list[i]:
                        different = True

                f_out.write(f'slot_pred: {[self.slot_label_lst[slot_id] for slot_id in slot_pred_id_list]}\n')
                if different:
                    f_out.write(f'slot_real: {[self.slot_label_lst[slot_id] for slot_id in slot_out_id_list]}\n')

                f_out.write(f'intent_pred: {self.intent_label_lst[intent_pred_id]}\n')
                if intent_pred_id != intent_out_id:
                    f_out.write(f'actual_real: {self.intent_label_lst[intent_out_id]}\n')

                f_out.write(f'\n')
