import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Function
from transformers import RobertaModel


# --------- this part is directly taken from https://github.com/bzantium/bert-DANN --------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base', from_tf=False, config=config)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat


class RobertaClassifier(nn.Module):

    def __init__(self, config):
        super(RobertaClassifier, self).__init__()
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.pooler(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out


class RobertaDomainClassifier(nn.Module):

    def __init__(self, config):
        super(RobertaDomainClassifier, self).__init__()
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, alpha):
        x = self.dropout(x)
        x = self.pooler(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = ReverseLayerF.apply(x, alpha)
        out = self.classifier(x)
        return out

# -------------- end of the excerpt from https://github.com/bzantium/bert-DANN -----------


class DomainAdversarialRoberta(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = RobertaEncoder(config=config)
        self.cls_classifier = RobertaClassifier(config=config)
        self.dom_classifier = RobertaDomainClassifier(config=config)

        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask,
                      input_ids_tgt, attention_mask_tgt,
                      labels, domains, alpha):
        features_src = self.encoder(input_ids, attention_mask)
        preds_src = self.cls_classifier(features_src)
        loss_cls = self.CELoss(preds_src, labels)

        # In the case we actually use data in the target domain (i.e. not at validation time)
        if input_ids_tgt is not None:
            features_tgt = self.encoder(input_ids_tgt, attention_mask_tgt)
            features_concat = torch.cat((features_src, features_tgt), 0)
            preds_dom = self.dom_classifier(features_concat, alpha)
            loss_dom = self.CELoss(preds_dom, domains)
            loss = loss_cls + loss_dom
        else:
            loss = loss_cls

        # Format outputs in a similar fashion to HuggingFace Transformers
        outputs = ModelOutput(loss=loss, logits=preds_src)
        return outputs


class ModelOutput:

    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits
