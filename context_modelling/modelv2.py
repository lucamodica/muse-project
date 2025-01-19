import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from torch import nn
from torch.functional import F
from transformers import RobertaTokenizer, RobertaModel

class AudioEncoder(nn.Module):
    def __init__(self, model_name, preprocessing=True, fine_tune=False, unfreeze_last_n_layers=2):
        """
        Generalized audio encoder for both encoder-only and encoder-decoder models.

        :param model_name: Hugging Face model name (e.g., "facebook/wav2vec2-base", "openai/whisper-small").
        :param preprocessing: Whether to apply preprocessing using a processor.
        :param fine_tune: Whether to fine-tune the model.
        :param unfreeze_last_n: Number of encoder layers to unfreeze if fine-tuning.
        """
        super(AudioEncoder, self).__init__()

        
        self.processor = AutoProcessor.from_pretrained(
            model_name) if preprocessing else None

        
        self.model = AutoModel.from_pretrained(model_name)

        
        self.is_encoder_decoder = hasattr(
            self.model, "encoder") and hasattr(self.model, "decoder")

        
        for param in self.model.parameters():
            param.requires_grad = False

        
        if fine_tune:
            if self.is_encoder_decoder:
                
                total_layers = len(self.model.encoder.layers)
                for layer_idx in range(total_layers - unfreeze_last_n_layers, total_layers):
                    for param in self.model.encoder.layers[layer_idx].parameters():
                        param.requires_grad = True
            elif hasattr(self.model, "encoder"):
                
                total_layers = len(self.model.encoder.layers)
                for layer_idx in range(total_layers - unfreeze_last_n_layers, total_layers):
                    for param in self.model.encoder.layers[layer_idx].parameters():
                        param.requires_grad = True
            else:
                print(
                    f"Warning: Fine-tuning is not supported for {model_name}")

    def preprocess_audio(self, audios, sampling_rate=16000):
        """
        Preprocess audio using the processor, if available.

        :param audios: Tensor of raw audio waveforms (shape: [batch_size, num_samples]).
        :param sampling_rate: Sampling rate of the audio (default: 16 kHz).
        :return: Preprocessed inputs for the model.
        """
        if self.processor:
            
            inputs = self.audio_processor(
                audios.squeeze(0),  
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
            if "input_values" in inputs:
                
                inputs = inputs.input_values[0]
            elif "input_features" in inputs:
                
                inputs = inputs.input_features[0]
            else:
                raise ValueError("Unsupported processor output format.")
        else:
            inputs = audios

        return inputs

    def forward(self, audio):
        """
        Forward pass to process audio and extract embeddings.

        :param audios: Tensor of raw audio waveforms (shape: [batch_size, num_samples]).
        :return: Mean-pooled audio embeddings (shape: [batch_size, hidden_size]).
        """
        
        audio_input = self.preprocess_audio(audio)
        print(audio_input.shape)

        
        

        
        with torch.no_grad() if not self.training or not any(
            p.requires_grad for p in self.model.parameters()
        ) else torch.enable_grad():
            if self.is_encoder_decoder:
                
                outputs = self.model.encoder(audio_input)
            elif hasattr(self.model, "encoder"):
                
                outputs = self.model(audio_input)
            else:
                raise ValueError(
                    f"Model {self.model.config.model_type} is not supported.")
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
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
        self.model = AutoModel.from_pretrained(model_name).to(device)

        
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
        return output, (hn, cn)  

class MultimodalClassifierWithLSTM(nn.Module):
    def __init__(self, audio_encoder, text_encoder, fusion_lstm, audio_lstm=None, text_lstm=None, hidden_dim=256, num_emotions=7, num_sentiments=3, max_utt=50, alternating=False):
        super(MultimodalClassifierWithLSTM, self).__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.fusion_lstm = fusion_lstm  
        self.audio_lstm = audio_lstm
        self.text_lstm = text_lstm
        self.emotion_head = nn.Linear(hidden_dim*2, num_emotions)
        self.sentiment_head = nn.Linear(hidden_dim*2, num_sentiments)
        self.max_utt = max_utt
        self.alternating = alternating
    
    def forward(self, audio_dialog_utts, text_dialog_utts, alternating=False):
       
        audio_emb = [self.audio_encoder(utt) for utt in audio_dialog_utts]
        audio_emb = torch.stack(audio_emb)

        text_emb = [self.text_encoder(utt) for utt in text_dialog_utts]
        text_emb = torch.stack(text_emb)  

        
        audio_emb = audio_emb.squeeze(1)
        text_emb = text_emb.squeeze(1)

        if self.alternating:
            
            audio_emb = audio_emb.unsqueeze(0)
            text_emb = text_emb.unsqueeze(0)
            
            
            padded_audio_emb = F.pad(audio_emb, (0, 0, 0, self.max_utt - audio_emb.size(1)))
            padded_text_emb = F.pad(text_emb, (0, 0, 0, self.max_utt - text_emb.size(1)))

            audio_lstm_out, (audio_hn, audio_cn) = self.audio_lstm(padded_audio_emb)
            text_lstm_out, (text_hn, text_cn) = self.text_lstm(padded_text_emb)

            audio_lstm_out = audio_lstm_out.squeeze(0)
            text_lstm_out = text_lstm_out.squeeze(0)

            audio_emotion_logits = self.emotion_head(audio_lstm_out)
            audio_sentiment_logits = self.sentiment_head(audio_lstm_out)

            text_emotion_logits = self.emotion_head(text_lstm_out)
            text_sentiment_logits = self.sentiment_head(text_lstm_out)

            return audio_emotion_logits, audio_sentiment_logits, text_emotion_logits, text_sentiment_logits
        
        
        fused_emb = torch.cat([audio_emb, text_emb], dim=-1)  

        
        fused_emb = fused_emb.unsqueeze(0)  

        padded_fused_emb = F.pad(fused_emb, (0, 0, 0, self.max_utt - fused_emb.size(1)))
        
        
        lstm_out, (hn, cn) = self.fusion_lstm(padded_fused_emb)

        

        lstm_out = lstm_out.squeeze(0)

        

        emotion_logits = self.emotion_head(lstm_out)
        sentiment_logits = self.sentiment_head(lstm_out)

        return emotion_logits, sentiment_logits

        
