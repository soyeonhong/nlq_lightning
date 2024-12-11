import torch
import random
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead


class GroundVQA(nn.Module):
    def __init__(self, 
                 lm_path, 
                 input_dim, 
                 
                 object_qa = False,
                 nlq_from_qa = False,
                 lm_weight = 0.2,
                 time_weight = 0.5,
                 
                 object_aug = False,
                 
                 debug = False,
                 freeze_word=False, 
                 max_v_len=256):
        super().__init__()
        self.object_qa = object_qa
        self.nlq_from_qa = nlq_from_qa
        self.lm_weight = lm_weight
        self.time_weight = time_weight
        
        self.object_aug = object_aug
        self.debug = debug

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

    def forward(self, 
                    v_feat, 
                    v_mask, 
                    q_token, 
                    q_mask, 
                    v_len=None,
                    
                    gt_segments=None, 
                    gt_labels=None,
                    labels=None, 
                    training=True,
                    
                    # object_qa
                    q_text_obj_pos=None,
                    q_token_obj_pos=None,
                    q_mask_obj_pos=None,
                    q_text_obj_neg=None,
                    q_token_obj_neg=None,
                    q_mask_obj_neg=None,
                    labels_obj_pos=None,
                    labels_obj_neg=None,
                    v_feat_for_obj=None,
                    v_mask_for_obj=None,
                    
                    # object_qug
                    q_token_obj_aug=None,
                    q_mask_obj_aug=None,
                    gt_segments_jit=None,
                    gt_labels_jit=None,
                    
                    **remains):
        
        if self.object_aug and training:
            if random.randint(0, 1) == 1: # change query from augmented queries
                
                num_sen = q_token_obj_aug.shape[1]
                random_sen = random.randint(0, num_sen-1)
                q_token = q_token_obj_aug[:, random_sen] # [B, L]
                q_mask = q_mask_obj_aug[:, random_sen] # [B]
            
        # encoder
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask) 
        
        # localizer
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
        
        if training and self.object_aug:
            gt_segments = gt_segments_jit
            gt_labels = gt_labels_jit
        
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
            
            if self.object_qa:
                if v_feat_for_obj is not None:
                    v_feat = v_feat_for_obj
                    v_mask = v_mask_for_obj
                
                # pos, neg selection
                if random.randint(0, 1) == 1 or self.debug: # positive
                    q_token = q_token_obj_pos
                    q_mask = q_mask_obj_pos
                    labels = labels_obj_pos
                else: # negative
                    q_token = q_token_obj_neg
                    q_mask = q_mask_obj_neg
                    labels = labels_obj_neg

                # encoder
                encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask) 
                
            outputs = self.lm(
                encoder_outputs=(encoder_out,),
                attention_mask=mask,
                labels=labels,
                output_hidden_states=False,
                output_attentions=False
            )
            lm_loss = outputs.loss
            
            log_dict = {
                'time_loss': time_loss.detach(),
                'lm_loss': lm_loss.detach()
            }
            
            if self.nlq_from_qa:
                torch.cuda.empty_cache()
                lm_logits = outputs.logits # [B, L, V]
                probs = torch.softmax(lm_logits, dim=-1) # [B, L, V]
                pred_tokens = torch.argmax(probs, dim=-1) # [B, L]
                qa_mask = torch.all(labels == pred_tokens, dim=1) # [B]
                qa_mask = torch.nonzero(qa_mask)
                
                if not qa_mask.shape[0] == 0:
                    encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
                    
                    selected_encoder_out_v = encoder_out_v[qa_mask].squeeze(1)
                    selected_v_mask = v_mask[qa_mask].squeeze(1)
                    selected_gt_segments = gt_segments[qa_mask].squeeze(1)
                    selected_gt_labels = gt_labels[qa_mask].squeeze(1)

                    nlq_results = self.nlq_head(
                        feat=selected_encoder_out_v.permute(0, 2, 1),  # (B, D, T)
                        mask=selected_v_mask.unsqueeze(1),  # (B, 1, T)
                        gt_segments=selected_gt_segments if training else None,
                        gt_labels=selected_gt_labels if training else None,
                        training=training,
                        v_lens=v_len if not training else None,
                    )
                    time_loss2 = nlq_results['final_loss']
                else:
                    time_loss2 = 0.0
                
                total_loss = 0.5 * time_loss + self.time_weight * time_loss2 + self.lm_weight * lm_loss
                log_dict['time_loss2'] = time_loss2
            else:
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
