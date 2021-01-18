#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers.modeling_bert import BertEncoder, BertPreTrainedModel
from transformers import PretrainedConfig
from torch.utils.data import Dataset
import numpy as np


class SubDataset(Dataset):
    def __init__(self, evaluate, total, base_name, num_parts, shape):
        self.evaluate = evaluate
        self.total = total
        self.base_name = base_name
        self.num_parts = num_parts
        self.shape = shape
        self.current_part = 1
        self.passed_index = 0
        self.dataset = None
        self._read_data()

    def _read_data(self):
        del self.dataset
        features = torch.load('{}_sub{}'.format(self.base_name, self.current_part))
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_gat_mask = [f.gat_mask for f in features]
        all_base_index = [f.base_index for f in features]
        all_tag_to_token = [f.tag_to_token_index for f in features]

        if self.evaluate:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index,
                                        all_cls_index, all_p_mask, gat_mask=all_gat_mask, base_index=all_base_index,
                                        tag2tok=all_tag_to_token, shape=self.shape,
                                        training=False)
        else:
            all_answer_tid = torch.tensor([f.answer_tid for f in features], dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids,
                                        all_answer_tid, all_start_positions, all_end_positions,
                                        all_cls_index, all_p_mask, gat_mask=all_gat_mask, base_index=all_base_index,
                                        tag2tok=all_tag_to_token, shape=self.shape,
                                        training=True)

    def __getitem__(self, index):
        if index - self.passed_index < 0:
            self.passed_index = 0
            self.current_part = 1
            self._read_data()
        elif index - self.passed_index >= len(self.dataset):
            self.passed_index += len(self.dataset)
            self.current_part += 1
            self._read_data()
        return self.dataset[index - self.passed_index]

    def __len__(self):
        return self.total


class StrucDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, gat_mask=None, base_index=None, tag2tok=None, shape=None, training=True):
        tensors = tuple(tensor for tensor in tensors if tensor is not None)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors
        self.gat_mask = gat_mask
        self.base_index = base_index
        self.tag2tok = tag2tok
        self.shape = shape
        self.training = training

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]

        gat_mask = np.load(self.gat_mask[index])
        gat_mask = torch.tensor(gat_mask, dtype=torch.long)
        output.append(gat_mask)

        tag_to_token_index = self.tag2tok[index]
        pooling_matrix = np.zeros(self.shape, dtype=np.double)
        for i in range(len(tag_to_token_index)):
            temp = tag_to_token_index[i]
            pooling_matrix[i][temp[0]: temp[1] + 1] = 1 / (temp[1] - temp[0] + 1)
        pooling_matrix = torch.tensor(pooling_matrix, dtype=torch.float)
        output.append(pooling_matrix)

        return tuple(item for item in output)

    def __len__(self):
        return len(self.tensors[0])


class GraphHtmlConfig(PretrainedConfig):
    def __init__(self,
                 args,
                 **kwargs):
        super().__init__(**kwargs)
        self.method = args.method
        self.model_type = args.model_type
        self.num_hidden_layers = args.num_node_block


class Link(nn.Module):
    def __init__(self, method='base'):
        super().__init__()
        self.method = method

    def forward(self, inputs, tag_to_token, gat_mask):
        assert tag_to_token.dim() == 3
        modified_tag2token = self.deduce_direct_string(tag_to_token)
        modified_gat_mask = self.deduce_child(gat_mask)
        outputs = torch.matmul(modified_tag2token, inputs)
        if modified_gat_mask is not None:
            for i in range(outputs.size(1) - 1, -1, -1):
                outputs[:, i] = torch.matmul(modified_gat_mask[:, i].unsqueeze(dim=1), outputs).squeeze(dim=1)
        return outputs

    def deduce_direct_string(self, tag_to_token):
        if self.method not in ['init_direct', 'init_child']:
            return tag_to_token
        temp = torch.zeros_like(tag_to_token)
        temp[tag_to_token > 0] = 1
        for i in range(tag_to_token.size(1)):
            temp[:, i] -= temp[:, i + 1:].sum(dim=1)
        temp[temp <= 0] = 0.
        for i in range(tag_to_token.size(1)):
            temp[:, i] /= temp[:, i].sum(dim=1, keepdim=True)
        return temp

    def deduce_child(self, gat_mask):
        if self.method != 'init_child':
            return None
        assert gat_mask.dim() == 3
        child = torch.zeros_like(gat_mask)
        l = gat_mask.size(1)
        for i in range(l):
            child[:, i] = gat_mask[:, i]
            for j in range(i + 1, l):
                temp = child[:, j].unsqueeze(dim=1) * gat_mask[:, j]
                child = ((child - temp) > 0).to(child.dtype)
        return child


class GraphHtmlBert(BertPreTrainedModel):
    def __init__(self, PTMForQA, config: GraphHtmlConfig):
        super(GraphHtmlBert, self).__init__(config)
        self.method = config.method
        self.base_type = config.model_type
        if config.model_type == 'bert':
            self.ptm = PTMForQA.bert
        elif config.model_type == 'albert':
            self.ptm = PTMForQA.albert
        elif config.model_type == 'electra':
            self.ptm = PTMForQA.electra
        else:
            raise NotImplementedError()
        self.link = Link(self.method)
        self.num_gat_layers = config.num_hidden_layers
        self.gat = BertEncoder(config)
        self.qa_outputs = PTMForQA.qa_outputs
        self.gat_outputs = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            gat_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            answer_tid=None,
            tag_to_tok=None,
    ):

        outputs = self.ptm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        outputs = outputs[2:]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,) + outputs

        gat_inputs = self.link(sequence_output, tag_to_tok, gat_mask)
        if head_mask is None:
            head_mask = [None] * self.num_gat_layers
        extended_gat_mask = gat_mask[:, None, :, :]
        gat_outputs = self.gat(gat_inputs, attention_mask=extended_gat_mask, head_mask=head_mask)
        final_outputs = gat_outputs[0]
        tag_logits = self.gat_outputs(final_outputs)
        tag_logits = tag_logits.squeeze(-1)
        outputs = (tag_logits,) + outputs

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        if answer_tid is not None:
            # If we are on multi-GPU, split add a dimension
            if len(answer_tid.size()) > 1:
                answer_tid = answer_tid.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = tag_logits.size(1)
            answer_tid.clamp_(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = loss_fct(tag_logits, answer_tid)
            outputs = (loss,) + outputs

        return outputs  # (loss), (total_loss), tag_logits, start_logits, end_logits, (hidden_states), (attentions)
