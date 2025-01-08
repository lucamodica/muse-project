import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoProcessor, BertTokenizer, DistilBertModel, DistilBertTokenizer
from torch.functional import F


class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", 
                 target_sr=16000, 
                 fine_tune=False, 
                 unfreeze_last_n=2,  # Number of last layers to unfreeze
                 device='cuda'):
        super(AudioEncoder, self).__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.target_sr = target_sr

        # Freeze entire model
        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune:
            # Unfreeze only the last `unfreeze_last_n` encoder layers
            total_layers = len(self.model.encoder.layers)
            for layer_idx in range(total_layers - unfreeze_last_n, total_layers):
                for param in self.model.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True

    def forward(self, waveforms):
        """
        :param waveforms: Tensor of shape [B, T] (already at self.target_sr)
        :return: A tensor of shape [B, hidden_dim] (audio embeddings for the batch)
        """
        # Ensure the waveforms are on the correct device
        waveforms = waveforms.to(self.model.device)
        
        # Prepare inputs for Wav2Vec2Processor
        inputs = self.processor(
            waveforms.cpu(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        ).input_values.squeeze((0, 1)).to(self.model.device)
    
        # Forward pass
        with torch.no_grad() if not self.training or not any(
            p.requires_grad for p in self.model.parameters()
        ) else torch.enable_grad():
            outputs = self.model(inputs)
            hidden_states = outputs.last_hidden_state  # shape [B, T, D]
    
        # Average pooling
        audio_emb = hidden_states.mean(dim=1)  # shape [B, D]
    
        return audio_emb

class TextEncoder(nn.Module):
    """
    Encodes text using a pretrained BERT model from Hugging Face.
    """

    def __init__(self,
                 model_name="distilbert-base-uncased",
                 fine_tune=False,
                 unfreeze_last_n_layers=2,
                 device='cuda'):
        super(TextEncoder, self).__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # If we're not fine-tuning at all, freeze everything.
        # Otherwise, freeze everything first, then selectively unfreeze layers.
        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune and unfreeze_last_n_layers > 0:
            # For BERT-base, there are 12 encoder layers: encoder.layer[0] ... encoder.layer[11].
            # Unfreeze the last N layers:
            total_layers = len(self.model.transformer.layer)
            for layer_idx in range(total_layers - unfreeze_last_n_layers, total_layers):
                for param in self.model.transformer.layer[layer_idx].parameters():
                    param.requires_grad = True

            # Optionally unfreeze the pooler layer if you use it
            #for param in self.model.pooler.parameters():
             #   param.requires_grad = True

    def forward(self, text_list):
        """
        :param text_list: A list of strings (or a single string) to encode.
        :return: A tensor of shape [batch_size, hidden_dim] with text embeddings
        """
        device = self.model.device

        # If a single string is passed, wrap it into a list
        if isinstance(text_list, str):
            text_list = [text_list]

        encodings = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)

        # If all parameters are frozen (no grad), then no_grad() is fine.
        # But if some layers are unfrozen, we want torch.enable_grad()
        # so that backprop can proceed for those layers.
        use_grad = any(p.requires_grad for p in self.model.parameters())
        with torch.enable_grad() if use_grad else torch.no_grad():
            outputs = self.model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask
            )

        # outputs.last_hidden_state -> shape [batch_size, seq_len, hidden_dim]
        # Typically we use the [CLS] token embedding as a single representation
        cls_emb = outputs.last_hidden_state[:, 0, :]  # shape [B, hidden_dim]
        return cls_emb

class MultimodalClassifier(nn.Module):
    def __init__(self,
                 fusion_dim=1536, 
                 audio_model_name="facebook/wav2vec2-base",
                 text_model_name="distilbert-base-uncased",
                 audio_fine_tune=False,
                 text_fine_tune=False,
                 unfreeze_last_n_audio=2,
                 unfreeze_last_n_text=2,
                 audio_encoder=None,
                 num_emotions=7,
                 num_sentiments=3):
        super(MultimodalClassifier, self).__init__()

        # Build encoders
        self.text_model_name = text_model_name
        
        if audio_encoder:
            self.audio_encoder = audio_encoder
            self.audio_encoder_name = audio_encoder.__class__.__name__
        else:
            self.audio_encoder = AudioEncoder(model_name=audio_model_name, fine_tune=audio_fine_tune, unfreeze_last_n=unfreeze_last_n_audio)
            self.audio_model_name = audio_model_name
            
        self.text_encoder = TextEncoder(model_name=text_model_name, fine_tune=text_fine_tune, unfreeze_last_n_layers=unfreeze_last_n_text)

        # If Wav2Vec2 base has 768 dims and BERT base has 768 dims -> total is 1536
        # If you use average pooling / CLS, that might remain 768 for each

        self.e_shared_classifier = nn.Linear(fusion_dim, num_emotions)
        self.s_shared_classifier = nn.Linear(fusion_dim, num_sentiments)

    def forward(self, audio_array, text):
        """
        :param waveform: Tensor of shape [1, T] or [B, 1, T] with audio waveforms
        :param sample_rate: int or list of ints (for multiple samples)
        :param text: list of strings or a single string
        :return: logits of shape [B, num_classes]
        """

        audio_array = audio_array.unsqueeze(1)
        audio_emb = self.audio_encoder(audio_array)  # shape [1, D]
        audio_emb = F.adaptive_avg_pool2d(audio_emb, 1)
        audio_emb = torch.flatten(audio_emb, 1)

        text_emb = self.text_encoder(text)  # shape [B, D]

        return audio_emb, text_emb