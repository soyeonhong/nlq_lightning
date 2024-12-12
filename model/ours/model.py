import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead
from model.ours.egovlp.model import TextOnlyFrozenInTime
from einops import rearrange

def freeze_module(module):
    for name, param in module.named_parameters():
        param.requires_grad = False

class GroundVQA(nn.Module):
    def __init__(self, lm_path, 
                 input_dim,
                 
                 # obj_mask
                 use_egovlp=False,
                 egovlp_path=None, 
                 max_obj_in_cap=None,
                 max_cap_len=None,
                 topk=None,
                 use_matmul=False,
                 use_sim=False,
                 
                 freeze_word=False, max_v_len=256):
        super().__init__()
        self.use_egovlp = use_egovlp
        self.egovlp_path = egovlp_path
        self.max_obj_in_cap = max_obj_in_cap
        self.max_cap_len = max_cap_len
        self.topk = topk
        self.use_matmul = use_matmul
        self.use_sim = use_sim

        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        self.lm: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(lm_path, local_files_only=True)

        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.lm_proj = nn.Linear(input_dim, lm_dim)
        self.v_emb = nn.Parameter(torch.randn((1, 1, lm_dim)))
        if freeze_word:
            for name, param in self.lm.named_parameters():
                if 'shared' in name:
                    param.requires_grad = False

        self.nlq_head = NLQHead(in_dim=lm_dim, max_v_len=max_v_len)
        
        if self.use_egovlp:
            self.egovlp = TextOnlyFrozenInTime(ckpt_path=self.egovlp_path)
            freeze_module(self.egovlp)
            
    def forward(self, 
                v_feat, 
                v_mask, 
                q_token, 
                q_mask, 
                
                # obj_mask
                obj_list_token=None,
                q_obj_token=None,
                
                gt_segments=None, 
                gt_labels=None, 
                labels=None, 
                v_len=None, 
                compute_loss=False, 
                training=True, 
                **remains):
        # compute obj_mask
        if self.use_egovlp:
            B = v_feat.shape[0]
            with torch.no_grad():
                obj_feat = self.egovlp(obj_list_token) # [B * C * O, D]
                query_feat = self.egovlp(q_obj_token) # [B, D]
            
            if not obj_feat.shape[0] == B * self.max_cap_len * self.max_obj_in_cap:
                obj_feat = obj_feat[:B * self.max_cap_len * self.max_obj_in_cap]
            obj_feat = rearrange(obj_feat, '(B C O) D -> B C O D', 
                                 B=B, C=self.max_cap_len, O=self.max_obj_in_cap, D=obj_feat.shape[-1]) # [B, C, O, D]
            obj_feat = torch.topk(obj_feat, self.topk, dim=2).indices # [B, C, k, D]
            obj_feat = obj_feat.float().mean(dim=2) # [B, C, D]
            
            if self.use_matmul:
                attn = torch.bmm(obj_feat, query_feat.unsqueeze(2)).squeeze(2) # [B, C]
            elif self.use_sim:
                attn = F.cosine_similarity(obj_feat, query_feat.unsqueeze(1), dim=-1)  # [B, C]
            attn = F.softmax(attn, dim=-1) # [B, C]
            attn = F.interpolate(attn.unsqueeze(1), size=(v_mask.shape[1],), mode='nearest').squeeze(1).to(dtype=v_mask.dtype) # [B, T]
            v_mask = attn
          
        # Encoder
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask)
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
        
        # Localizer
        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            gt_segments=gt_segments if training else None,
            gt_labels=gt_labels if training else None,
            training=training,
            v_lens=v_len if not training else None
        )
        
        if training:
            output_dict = {}
            # Compute losses
            time_loss = nlq_results['final_loss'] * 1.0
            outputs = self.lm(
                encoder_outputs=(encoder_out,),
                attention_mask=mask,
                labels=labels,
            )
            lm_loss = outputs.loss
            log_dict = {
                'time_loss': time_loss.detach(),
                'lm_loss': lm_loss.detach()
            }
            
            total_loss = 0.5 * time_loss + 0.5 * lm_loss
            
            log_dict.update({'total_loss': total_loss.detach()})
            output_dict.update({'loss': total_loss})
            output_dict.update({'log_dict': log_dict})
            
            return output_dict
        else:
            # Generate answer tokens
            answer_tokens = self.lm.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
                attention_mask=mask,
                max_new_tokens=32
            )
            return nlq_results, answer_tokens

    def forward_encoder(self, v_feat, v_mask, q_token, q_mask):
        B, L, D = v_feat.shape
        v_feat = self.lm_proj(v_feat)
        v_feat = v_feat + self.v_emb.expand((B, L, -1))
        q_feat = self.lm.encoder.embed_tokens(q_token)
        lm_input = torch.cat([q_feat, v_feat], dim=1)
        lm_mask = torch.cat([q_mask, v_mask], dim=1)
        out = self.lm.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask
        )
        return out.last_hidden_state, lm_mask
