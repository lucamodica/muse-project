import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, DistilBertTokenizer
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

class FusionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1, bidirectional=False, dropout=0.1):
        super(FusionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        returns: (batch_size, seq_len, hidden_dim)
        """
        output, (hn, cn) = self.lstm(x)
        return output, (hn, cn)  # output is the sequence of hidden states

class MultimodalClassifierWithLSTM(nn.Module):
    def __init__(self, audio_encoder, text_encoder, fusion_lstm, hidden_dim=256, num_emotions=7, num_sentiments=3, max_utt=50):
        super(MultimodalClassifierWithLSTM, self).__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.fusion_lstm = fusion_lstm  # e.g., an instance of FusionLSTM
        self.emotion_head = nn.Linear(hidden_dim*2, num_emotions)
        self.sentiment_head = nn.Linear(hidden_dim*2, num_sentiments)
        self.max_utt = max_utt
    
    def forward(self, audio_dialog_utts, text_dialog_utts):
       
        audio_emb = self.audio_encoder(audio_dialog_utts)
        audio_emb = F.adaptive_avg_pool2d(audio_emb, 1)
        audio_emb = torch.flatten(audio_emb, 1)

        text_emb = [self.text_encoder(utt) for utt in text_dialog_utts]
        text_emb = torch.stack(text_emb)  # shape (B, hidden_dim)

        text_emb = text_emb.squeeze(1)


        #print("Final Audio emb shape: ", audio_emb.shape)
        #print("Final Text emb shape: ", text_emb.shape)
        
        # Combine
        fused_emb = torch.cat([audio_emb, text_emb], dim=-1)  # (utts, audio_enc_dim + text_enc_dim)

        
        # Reshape back to (B, S, fused_dim) for the LSTM
        fused_emb = fused_emb.unsqueeze(0)  # (1, utts, audio_enc_dim + text_enc_dim)

        padded_fused_emb = F.pad(fused_emb, (0, 0, 0, self.max_utt - fused_emb.size(1)))
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.fusion_lstm(padded_fused_emb)

        #print("LSTM out shape: ", lstm_out.shape)

        lstm_out = lstm_out.squeeze(0)

        # Classification heads

        emotion_logits = self.emotion_head(lstm_out)
        sentiment_logits = self.sentiment_head(lstm_out)

        return emotion_logits, sentiment_logits

        
