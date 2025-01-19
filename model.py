import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoProcessor, BertTokenizer, DistilBertModel, DistilBertTokenizer, RobertaTokenizer, RobertaModel
from torch.functional import F


class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", 
                 target_sr=16000, 
                 fine_tune=False, 
                 unfreeze_last_n=2,  
                 device='cuda'):
        super(AudioEncoder, self).__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.target_sr = target_sr

        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune:
            total_layers = len(self.model.encoder.layers)
            for layer_idx in range(total_layers - unfreeze_last_n, total_layers):
                for param in self.model.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True

    def forward(self, waveforms):
        """
        :param waveforms: Tensor of shape [B, T] (already at self.target_sr)
        :return: A tensor of shape [B, hidden_dim] (audio embeddings for the batch)
        """
        waveforms = waveforms.to(self.model.device)
        
        inputs = self.processor(
            waveforms.cpu(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        ).input_values.squeeze((0, 1)).to(self.model.device)
    
        with torch.no_grad() if not self.training or not any(
            p.requires_grad for p in self.model.parameters()
        ) else torch.enable_grad():
            outputs = self.model(inputs)
            hidden_states = outputs.last_hidden_state 
    
        audio_emb = hidden_states.mean(dim=1)
    
        return audio_emb

class TextEncoder(nn.Module):
    """
    Encodes text using a pretrained RoBERTa model from Hugging Face.
    """

    def __init__(self,
                 model_name="roberta-base",
                 fine_tune=False,
                 unfreeze_last_n_layers=2,
                 device='cuda'):
        super(TextEncoder, self).__init__()
        
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        
        self.model = RobertaModel.from_pretrained(model_name).to(device)

        
        for param in self.model.parameters():
            param.requires_grad = False

        
        if fine_tune and unfreeze_last_n_layers > 0:
            total_layers = len(self.model.encoder.layer)  
            for layer_idx in range(total_layers - unfreeze_last_n_layers, total_layers):
                for param in self.model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True

    def forward(self, text_list):
        """
        :param text_list: A list of strings (or a single string) to encode.
        :return: A tensor of shape [batch_size, hidden_dim] with text embeddings
        """
        device = self.model.device

        
        if isinstance(text_list, str):
            text_list = [text_list]

        encodings = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)

        
        
        
        use_grad = any(p.requires_grad for p in self.model.parameters())
        with torch.enable_grad() if use_grad else torch.no_grad():
            outputs = self.model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask
            )

        
        
        cls_emb = outputs.last_hidden_state[:, 0, :]  
        return cls_emb

class MultimodalClassifier(nn.Module):
    def __init__(self,
                 fusion_dim=1536, 
                 alternating=False,
                 audio_model_name="facebook/wav2vec2-base",
                 text_model_name="roberta-base",
                 audio_fine_tune=False,
                 text_fine_tune=False,
                 unfreeze_last_n_audio=2,
                 unfreeze_last_n_text=2,
                 audio_encoder=None,
                 num_emotions=7,
                 num_sentiments=3):
        super(MultimodalClassifier, self).__init__()

        
        self.text_model_name = text_model_name
        
        if audio_encoder:
            self.audio_encoder = audio_encoder
            self.audio_encoder_name = audio_encoder.__class__.__name__
        else:
            self.audio_encoder = AudioEncoder(model_name=audio_model_name, fine_tune=audio_fine_tune, unfreeze_last_n=unfreeze_last_n_audio)
            self.audio_model_name = audio_model_name
            
        self.text_encoder = TextEncoder(model_name=text_model_name, fine_tune=text_fine_tune, unfreeze_last_n_layers=unfreeze_last_n_text)

        
        

        
        

        self.emotion_classifier = nn.Linear(fusion_dim, num_emotions)
        self.sentiment_classifier = nn.Linear(fusion_dim, num_sentiments)

    def forward(self, audio_array, text):
        """
        :param waveform: Tensor of shape [1, T] or [B, 1, T] with audio waveforms
        :param sample_rate: int or list of ints (for multiple samples)
        :param text: list of strings or a single string
        :return: logits of shape [B, num_classes]
        """

        audio_array = audio_array.unsqueeze(1)
        audio_emb = self.audio_encoder(audio_array)  
        audio_emb = F.adaptive_avg_pool2d(audio_emb, 1)
        audio_emb = torch.flatten(audio_emb, 1)

        text_emb = self.text_encoder(text)  

        return audio_emb, text_emb