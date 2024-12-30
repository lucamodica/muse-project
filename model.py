import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model
import librosa

class MultimodalClassifier(nn.Module):
    def __init__(self,
                 audio_model_name="facebook/wav2vec2-base",
                 text_model_name="bert-base-uncased",
                 audio_fine_tune=False,
                 text_fine_tune=False,
                 hidden_dim=768,
                 num_classes=4):
        """
        :param audio_model_name: e.g., "facebook/wav2vec2-base"
        :param text_model_name: e.g., "bert-base-uncased"
        :param audio_fine_tune: whether to fine-tune the audio encoder
        :param text_fine_tune: whether to fine-tune the text encoder
        :param hidden_dim: dimension of the embeddings (depends on the model)
        :param num_classes: number of emotion classes (e.g., 4 for IEMOCAP: angry, happy, sad, neutral)
        """
        super(MultimodalClassifier, self).__init__()

        # Build encoders
        self.audio_encoder = AudioEncoder(model_name=audio_model_name, fine_tune=audio_fine_tune)
        self.text_encoder = TextEncoder(model_name=text_model_name, fine_tune=text_fine_tune)

        # If Wav2Vec2 base has 768 dims and BERT base has 768 dims -> total is 1536
        # If you use average pooling / CLS, that might remain 768 for each
        self.fusion_dim = hidden_dim * 2

        # A simple linear head for classification
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, waveform, sample_rate, text):
        """
        :param waveform: Tensor of shape [1, T] or [B, 1, T] with audio waveforms
        :param sample_rate: int or list of ints (for multiple samples)
        :param text: list of strings or a single string
        :return: logits of shape [B, num_classes]
        """
        # 1) Audio embeddings
        if isinstance(sample_rate, int):
            # Single example
            audio_emb = self.audio_encoder(waveform, sample_rate)  # shape [1, D]
        else:
            # If you have a batch of waveforms with different sample rates, handle them accordingly
            raise NotImplementedError("Batch of waveforms with different sample rates not implemented in this simple example.")

        # 2) Text embeddings
        text_emb = self.text_encoder(text)  # shape [B, D]
        # If it's a single item, shape might be [1, D]

        # 3) Fuse (concat)
        fused = torch.cat([audio_emb, text_emb], dim=-1)  # shape [B, 2D]

        # 4) Classification
        logits = self.classifier(fused)  # shape [B, num_classes]

        return logits

class AudioEncoder(nn.Module):
    """
    Uses a Wav2Vec2 model to encode raw waveform into an audio embedding.
    """

    def __init__(self, model_name="facebook/wav2vec2-base", target_sr=16000, fine_tune=True):
        """
        :param model_name: Hugging Face model name, e.g., "facebook/wav2vec2-base".
        :param target_sr: Sample rate the model expects, typically 16k.
        :param fine_tune: Whether we allow gradient updates in the model.
        """
        super(AudioEncoder, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.target_sr = target_sr

        if not fine_tune:
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, waveform, sample_rate):
        """
        :param waveform: (channels, time)  e.g., shape [1, T]
        :param sample_rate: The actual sample rate of waveform
        :return: A single embedding vector of size [batch_size, hidden_dim] if you pass multiple waveforms
        """
        # If single waveform, shape might be [1, T]. For multi-batch, [B, 1, T].
        # We'll handle single vs. multiple in a simpler way here.

        # Resample if needed
        if sample_rate != self.target_sr:
            # Convert waveform to numpy, resample, then back to tensor
            waveform_np = waveform.squeeze().numpy()
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=self.target_sr)
            waveform = torch.tensor(waveform_np).unsqueeze(0)

        # If waveform is stereo or has multiple channels, average them
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = self.processor(waveform, sampling_rate=self.target_sr, return_tensors="pt")
        with torch.no_grad() if not self.training or not any(p.requires_grad for p in self.model.parameters()) else torch.enable_grad():
            outputs = self.model(inputs.input_values)
            # outputs.last_hidden_state -> shape (batch_size, seq_len, hidden_dim)

        # Simple average pooling across time
        # If you prefer [CLS], note that Wav2Vec2 doesn't have a CLS token in the same sense as BERT
        # so average pooling is common.
        hidden_states = outputs.last_hidden_state  # shape [B, T, D]
        audio_emb = hidden_states.mean(dim=1)      # shape [B, D]

        return audio_emb
class TextEncoder(nn.Module):
    """
    Encodes text using a pretrained BERT model from Hugging Face.
    """

    def __init__(self, model_name="bert-base-uncased", fine_tune=True):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, text_list):
        """
        :param text_list: A list of strings (or a single string) to encode.
        :return: A tensor of shape [batch_size, hidden_dim] with text embeddings
        """
        # If a single string is passed, wrap it into a list
        if isinstance(text_list, str):
            text_list = [text_list]

        encodings = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad() if not self.training or not any(p.requires_grad for p in self.model.parameters()) else torch.enable_grad():
            outputs = self.model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask
            )

        # outputs.last_hidden_state -> shape [batch_size, seq_len, hidden_dim]
        # Using [CLS] token embedding as a single representation
        cls_emb = outputs.last_hidden_state[:, 0, :]  # shape [B, hidden_dim]
        return cls_emb
