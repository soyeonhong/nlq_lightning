import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.utils.rnn import pad_sequence
from transformers.models.t5.modeling_t5 import T5Stack, T5EncoderModel

from model.ours.nlq_head import NLQHead

def _freeze_shared_parameters(model):
    """
    Freeze parameters with 'shared' in their name.
    """
    for name, param in model.named_parameters():
        if 'shared' in name:
            param.requires_grad = False

class GroundVQA(nn.Module):
    def __init__(self, lm_path, 
                       input_dim, 
                       
                       feedback=False,
                       feedback_w=0.5,
                       lm_proj_feed=False,
                       feedback_type='kl',
                       
                       scene_module=False,
                       mask_size=0,
                       scene_arch='encoder',
                       max_num_scene=1000000,

                       freeze_word=False, 
                       max_v_len=256):
        super().__init__()
        self.feedback = feedback
        self.feedback_w = feedback_w
        self.lm_proj_feed = lm_proj_feed
        self.feedback_type = feedback_type
        self.scene_module = scene_module
        self.mask_size = mask_size
        self.scene_arch = scene_arch
        self.max_num_scene = max_num_scene

        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        self.lm: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(lm_path, local_files_only=True)

        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.lm_proj = nn.Linear(input_dim, lm_dim)
        self.v_emb = nn.Parameter(torch.randn((1, 1, lm_dim)))
        if freeze_word:
            _freeze_shared_parameters(self.lm)

        self.nlq_head = NLQHead(in_dim=lm_dim, max_v_len=max_v_len)
        
        if self.scene_arch == 'encoder':
            config = self.lm.config
            config.num_layers = 3
            self.scene = T5EncoderModel(config)
            self.s_emb = nn.Parameter(torch.randn((1, 1, lm_dim)))
        elif self.scene_arch == 'decoder':
            config = self.lm.config      
            config.num_layers = 3
            config.is_decoder = True
            self.scene = T5Stack(config, self.lm.shared)
            
        if self.scene_module:      
            if freeze_word:
               _freeze_shared_parameters(self.scene)
               
    def forward(self, 
                v_feat, 
                v_mask, 
                q_token, 
                q_mask, 
                
                # scene_module
                scene=None,
                scene_mask=None,
                
                gt_segments=None, 
                gt_labels=None, 
                labels=None, 
                v_len=None, 
                compute_loss=False, 
                training=True, 
                **remains):
            
        # Encoder
        encoder_out, mask, sim_v_feat, hidden_states = self.forward_encoder(v_feat, v_mask, q_token, q_mask, scene, scene_mask)

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
            
            if self.feedback:
                feedback_loss = 0
                p = F.log_softmax(sim_v_feat, dim=-1) # desired distribution
                if self.lm_proj_feed:
                    sim_ori_v_feat = cosine_similarity(v_feat)
                    q = F.softmax(sim_ori_v_feat, dim=-1) # predicted distribution
                    # KL divergence
                    if self.feedback_type == 'kl':
                        feedback_loss = F.kl_div(p, q, reduction='batchmean')
                    # RMSE
                    elif self.feedback_type == 'rmse':
                        feedback_loss += F.mse_loss(sim_ori_v_feat, sim_v_feat)
                else:
                    for i in range(len(hidden_states)):
                        hidden = hidden_states[i][:, -v_feat.shape[1]:] # (B, T, D)
                        
                        # cosine similarity
                        sim_hidden = cosine_similarity(hidden)
                        
                        # KL divergence
                        q = F.softmax(sim_hidden, dim=-1) # predicted distribution
                        feedback_loss += F.kl_div(p, q, reduction='batchmean')
                
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

    def forward_encoder(self, v_feat, v_mask, q_token, q_mask, scene=None, scene_mask=None):
        B, L, D = v_feat.shape
        v_feat = self.lm_proj(v_feat)
        
        if self.feedback:
            sim_v_feat = cosine_similarity(v_feat)
        
        if self.scene_module:
            scene = self.lm_proj(scene) # (B, N, D)
            if self.scene_arch == 'encoder':
                scene = scene + self.s_emb.expand((B, scene.size(1), -1))
                scene_input = torch.cat([scene, v_feat], dim=1) # (B, T + N, D)
                scene_mask = torch.cat([scene_mask, v_mask], dim=1) # (B, T + N)
                
                scene_out = self.scene(
                    inputs_embeds=scene_input,
                    attention_mask=scene_mask).last_hidden_state # (B, T + N, D)
                
                v_feat = scene_out[:, -v_feat.shape[1]:] # (B, T, D)
            elif self.scene_arch == 'decoder':
                scene_out = self.scene(
                    inputs_embeds = v_feat,
                    attention_mask = v_mask,
                    encoder_hidden_states = scene,
                    encoder_attention_mask = scene_mask,
                ).last_hidden_state
                v_feat = scene_out # (B, T, D)
                torch.cuda.empty_cache()
            
        q_feat = self.lm.encoder.embed_tokens(q_token)

        v_feat = v_feat + self.v_emb.expand((B, L, -1)) # (B, T, D)
        q_feat = self.lm.encoder.embed_tokens(q_token)
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
        
def cosine_similarity(v_feat):
    """
    Computes the pairwise cosine similarity between vectors in the input tensor.

    Args:
        v_feat (torch.Tensor): Input tensor of shape (B, T, D), where:
                               - B is the batch size
                               - T is the sequence length
                               - D is the feature dimension

    Returns:
        torch.Tensor: Pairwise cosine similarity matrix of shape (B, T, T).
    """
    # Compute the inner product
    sim_v_feat = torch.bmm(v_feat, v_feat.transpose(1, 2))  # Shape: (B, T, T)

    # Compute the norm of each vector
    norm = torch.norm(v_feat, dim=-1, keepdim=True)  # Shape: (B, T, 1)

    # Normalize the inner product using the norms
    sim_v_feat = sim_v_feat / (norm @ norm.transpose(1, 2) + 1e-8)  # Shape: (B, T, T)

    return sim_v_feat