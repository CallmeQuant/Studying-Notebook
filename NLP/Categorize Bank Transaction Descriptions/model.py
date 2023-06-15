import torch
from torch import nn

class RobertaClassifier(nn.Module):
  def __init__(self, base_model, hidden_size, num_class, dropout = 0.5):
    super(RobertaClassifier, self).__init__()
    self.roberta = base_model
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()
    self.classifier = nn.Linear(4*hidden_size, num_class)

  def forward(self, input_ids, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None,
                start_positions = None, end_positions = None):
    outputs = self.roberta(input_ids,
                            attention_mask = attention_mask,
#                            token_type_ids=token_type_ids,
                            position_ids = position_ids,
                            head_mask = head_mask)

    hidden_states = outputs[2]
    pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
    pooled_output = pooled_output[:, 0, :]
    pooled_output = self.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    return logits
