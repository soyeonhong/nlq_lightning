import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead


class GroundVQA(nn.Module):
    def __init__(self, lm_path, 
                       input_dim, 
                       
                       feedback=False,
                       feedback_w=0.5,
                       lm_proj_feed=False,
                       lm_enc_feed=False,
                       feed_last=False,
                       feedback_type='kl',
                       feedback_mask=False,

                       freeze_word=False, 
                       max_v_len=256):
        super().__init__()
        self.feedback = feedback
        self.feedback_w = feedback_w
        self.lm_proj_feed = lm_proj_feed
        self.lm_enc_feed = lm_enc_feed
        self.feedback_type = feedback_type
        self.feed_last = feed_last
        self.feedback_mask = feedback_mask

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

    def forward(self, v_feat, v_mask, q_token, q_mask, gt_segments=None, gt_labels=None, 
            labels=None, v_len=None, compute_loss=False, training=True, **remains):
        # Encoder
        encoder_out, mask, sim_v_feat, hidden_states = self.forward_encoder(v_feat, v_mask, q_token, q_mask)
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
            
            if self.feedback and not self.feedback_mask:
                feedback_loss1 = feedback_loss2 = 0
                sim_ori_v_feat = cosine_similarity(v_feat)
                if self.lm_proj_feed:
                    feedback_loss1 = self.compute_feedback_loss(sim_ori_v_feat, sim_v_feat)
                if self.lm_enc_feed:
                    if self.feed_last:
                        hidden_states = hidden_states[-1]
                        hidden = hidden_states[:, -v_feat.shape[1]:] # (B, T, D)
                        sim_hidden = cosine_similarity(hidden) # (B, T, T)
                        feedback_loss2 = self.compute_feedback_loss(sim_ori_v_feat, sim_hidden)
                    else:
                        for i in range(len(hidden_states)):
                            hidden = hidden_states[i][:, -v_feat.shape[1]:] # (B, T, D)
                            sim_hidden = cosine_similarity(hidden) # (B, T, T)     
                            feedback_loss2 += self.compute_feedback_loss(sim_ori_v_feat, sim_hidden)
    
                feedback_loss = feedback_loss1 + feedback_loss2
                
                log_dict.update({'feedback_loss': feedback_loss.detach()})
                total_loss = 0.5 * time_loss + 0.25 * lm_loss + self.feedback_w * feedback_loss
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
        
        if self.feedback:
            sim_v_feat = cosine_similarity(v_feat)
            
        v_feat = v_feat + self.v_emb.expand((B, L, -1))
        q_feat = self.lm.encoder.embed_tokens(q_token)
        
        if self.feedback_mask:
            attn = torch.bmm(v_feat, q_feat.transpose(1, 2)) # (B, T_v, T_q)
            attn = attn.mean(dim=-1) # (B, T_v)
            attn = F.softmax(attn, dim=-1) # (B, T_v)
            v_mask = attn
            
        lm_input = torch.cat([q_feat, v_feat], dim=1)
        lm_mask = torch.cat([q_mask, v_mask], dim=1)
        out = self.lm.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask,
            output_hidden_states=True if self.feedback else None)

        if self.feedback:
            # exclude the first hidden state(=not a transformer output)
            return out.last_hidden_state, lm_mask, sim_v_feat, out.hidden_states[1: ] 
        else:
            return out.last_hidden_state, lm_mask, None, None
        
    def compute_feedback_loss(self, target, prediction):
        if self.feedback_type == 'kl':
            # KL divergence
            return F.kl_div(F.log_softmax(prediction, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')
        elif self.feedback_type == 'rmse':
            # RMSE
            return F.mse_loss(target, prediction)
        else:
            raise ValueError(f"Unsupported feedback type: {self.feedback_type}")

def cosine_similarity(v_feat):
    # Compute the inner product
    sim_v_feat = torch.bmm(v_feat, v_feat.transpose(1, 2))  # Shape: (B, T, T)

    # Compute the norm of each vector
    norm = torch.norm(v_feat, dim=-1, keepdim=True)  # Shape: (B, T, 1)

    # Normalize the inner product using the norms
    sim_v_feat = sim_v_feat / (norm @ norm.transpose(1, 2) + 1e-8)  # Shape: (B, T, T)

    return sim_v_feat