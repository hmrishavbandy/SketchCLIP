from collections import OrderedDict
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

from clip_sp import clip
from clip_sp.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.model_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe', 
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.n_ctx}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class MultiModalAdaptivePromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.input_shape[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.prompt_depth  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('Design: SketchCLIP')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        # self.proj = nn.Linear(ctx_dim, 768)
        # self.proj.half()
        
        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx_visual = torch.randn(n_ctx, 768, dtype=dtype).cuda()
        nn.init.normal_(self.ctx_visual, std=0.02)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.set_classifier = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, 3)),
            
        ]))
        
        if cfg.precision == "fp16":

            ctx_bias=torch.randn(3,ctx_dim).to(torch.float16).cuda()
            nn.init.normal_(ctx_bias, std=0.02)

            self.code_vectors=nn.Parameter(ctx_bias)
            self.code_vectors.requires_grad_(True)
            self.meta_net.half()
            self.set_classifier.half()
            self.code_vectors.half()
        else:
            ctx_bias=torch.randn(3,ctx_dim).cuda()
            nn.init.normal_(ctx_bias, std=0.02)

            self.code_vectors=nn.Parameter(ctx_bias)
            self.code_vectors.requires_grad_(True)
        
        

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.visual_prompts = nn.ParameterList([nn.Parameter(torch.randn(n_ctx, 768))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.visual_prompts:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def visual_prompt_generator(self):
        visual_deep_prompts = []
        
        return self.ctx_visual, self.visual_prompts


    def forward(self,im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        code_vectors=self.code_vectors
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        selection = self.set_classifier(im_features)

        bias_set = F.softmax(selection,dim=1)@code_vectors
        bias_set = bias_set.unsqueeze(1)
        bias = bias.unsqueeze(1)        # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)          # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias + bias_set          # (batch, n_ctx, ctx_dim)

        # print("{:.3f} {:.3f} {:.3f}".format(torch.norm(ctx),torch.norm(bias),torch.norm(bias_set)), end = " ")
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts, self.compound_prompts_text, selection 



def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if reduction == 'none':
            return batchloss
        elif reduction == 'mean':
            return torch.mean(batchloss)
        elif reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')
            

class CustomCLIP_MAPLE_Adaptive(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalAdaptivePromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        encoder_hidden_dim = 512
        decoder_hidden_dim = 512

        self.decoder = nn.GRU(5 + encoder_hidden_dim, decoder_hidden_dim, batch_first=True)
        self.linear_output = nn.Linear(decoder_hidden_dim, 5)


    def forward(self, image, label=None, label_set=None, training=True, i1=0, i2 =0, i3=0, points2=0, points3=0, seq_len2 =0, seq_len3 =0, is_singular = False,lamda = 0, to_mix = True, logits_only=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        visual_ctx, deep_compound_prompts_vision = self.prompt_learner.visual_prompt_generator()
        image_features = self.image_encoder(image.type(self.dtype), visual_ctx, deep_compound_prompts_vision)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        prompts, deep_compound_prompts_text, selection= self.prompt_learner(image_features)
        
        

        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts, deep_compound_prompts_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        if logits_only:
            return logits
        loss_set = F.cross_entropy(selection, label_set.long())

        if training and not is_singular:
            device="cuda"

            i2_features = self.image_encoder(i2.type(self.dtype), visual_ctx, deep_compound_prompts_vision) 
            i2_features = i2_features / i2_features.norm(dim=-1, keepdim=True)

            i3_features = self.image_encoder(i3.type(self.dtype), visual_ctx, deep_compound_prompts_vision) 
            i3_features = i3_features / i3_features.norm(dim=-1, keepdim=True)
            
            i_features=torch.concat([i2_features,i3_features]).type(self.dtype)
            i_features=i_features / i_features.norm(dim=-1, keepdim=True)

            
            points_= torch.concat([points2,points3]).type(self.dtype)

            target_coord = points_


            


            seq_len = torch.concat([seq_len2,seq_len3])

            decoder_input = torch.cat((i_features.unsqueeze(1).repeat(1, seq_len.max(), 1), target_coord[:,:-1, :]), dim=-1)
            decoder_input = pack_padded_sequence(decoder_input, seq_len.cpu().int(),  batch_first=True, enforce_sorted=False)
            output_hiddens, _ = self.decoder(decoder_input.float())
            output_hiddens, _ = pad_packed_sequence(output_hiddens, batch_first=True)

            output = self.linear_output(output_hiddens)

            output_XY, pen_bits = output.split([2, 3], dim=-1)
            output_XY = torch.clamp(output_XY,min=0.0,max = 1.0)

            # pen_bits_onehot = F.one_hot(pen_bits.argmax(-1), num_classes=3).float()
            
            mask = torch.ones(output.shape[:2]).to(device)
            target_cross_entropy = target_coord[:, 1:, 2:].argmax(axis=-1)
            for i_num, seq in enumerate(seq_len):
                mask[i_num, seq:] = 0.
                target_cross_entropy[i_num, seq:] = 10

            loss_coor = F.mse_loss(output_XY.double(), target_coord[:, 1:, :2].double(), reduction='none') * mask.unsqueeze(-1)

            loss_pen = F.cross_entropy(pen_bits.view(-1, pen_bits.shape[-1]).double(), target_cross_entropy.view(-1), ignore_index=10)

            loss_coor = loss_coor.sum() / mask.sum()

            loss = loss_coor + loss_pen

            if to_mix==True:

                ### Mixup implementation: 
                l1 = torch.zeros(len(i1),3).cuda()
                l2 = torch.zeros(len(i2),3).cuda()
                l3 = torch.zeros(len(i3),3).cuda()
                
                l1[:,0]=1
                l2[:,1]=1
                l3[:,2]=1
                
                
                i1_features = self.image_encoder(i1.type(self.dtype), visual_ctx, deep_compound_prompts_vision) 
                i1_features = i1_features / i1_features.norm(dim=-1, keepdim=True)

            
            
                mix_repr = i1_features*lamda[0].unsqueeze(-1).repeat(1,512)+i2_features*(lamda[1].unsqueeze(-1).repeat(1,512))+i3_features*(lamda[2].unsqueeze(-1).repeat(1,512))
                mix_labels = l1*lamda[0].unsqueeze(-1).repeat(1,3)+l2*(lamda[1].unsqueeze(-1).repeat(1,3))+l3*(lamda[2].unsqueeze(-1).repeat(1,3))
                
                
                output_labels =self.prompt_learner.set_classifier(mix_repr)
                loss_set_v2 = softmax_cross_entropy_with_softtarget(output_labels, mix_labels)
                loss_set = (loss_set+loss_set_v2)

        if is_singular:
            loss = 0
        
        if training:
            return F.cross_entropy(logits, label), loss_set, loss
        else:
            return logits, F.cross_entropy(logits, label), loss_set


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

