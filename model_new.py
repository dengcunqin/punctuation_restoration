import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# from data_module import sort_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import Encoder
import math

class Model_new(nn.Module):

	def __init__(self, params):
		super().__init__()
		self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
		self.encoder = Encoder(in_channels=params.embedding_dim,
								out_channels=params.embedding_dim,
								kernel_size=3,
								padding="same",
								n_layers=3)
		self.biGRU = nn.LSTM(params.embedding_dim, params.hidden_size1, 2, bidirectional=True, batch_first=True)
		self.GRU = nn.LSTM(params.hidden_size1*2, params.hidden_size2, 1, batch_first=True)
		self.decoder_punct = nn.Linear(params.hidden_size2*2, params.out_size_punct)
		self.dropout1 = nn.Dropout(params.dropout)
		self.dropout2 = nn.Dropout(params.dropout)
		self.config = params

	def forward(
		self, 
		input_token_ids,
		valid_ids=None,
		label_lens=None,
	):
		embedding_out = self.embedding(input_token_ids) # [batch_size, max_seq_length, embedding_dim]
		embedding_out = self.encoder(embedding_out)
		batch_size, max_seq_length, embedding_dim = embedding_out.shape
		# Placeholder for the output with the same shape as `embedding_out`
		valid_output = torch.zeros_like(embedding_out)
		# Create a mask for valid positions
		valid_mask = valid_ids.to(torch.bool)
		# Flatten the mask and the embedding output to map valid positions
		flat_valid_mask = valid_mask.view(-1)
		flat_embedding_out = embedding_out.view(-1, embedding_dim)
		# Filter out the valid embeddings
		valid_embeddings = flat_embedding_out[flat_valid_mask]
		# We need a cumulative sum of the valid_ids to determine the correct indices in valid_output
		cumulative_valid_counts = valid_ids.cumsum(dim=1) - 1
		# Flatten cumulative_valid_counts to use it for indexing
		flat_cumulative_valid_counts = cumulative_valid_counts.view(-1)
		# Use the cumulative sum as indices to place valid_embeddings in the valid_output
		# We also need a range for each example in the batch
		batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, max_seq_length).reshape(-1)
		batch_indices = batch_indices.to(flat_valid_mask.device)
		batch_indices = batch_indices[flat_valid_mask]  # Select only indices for valid embeddings
		# Now we place the valid embeddings into the correct positions in valid_output
		valid_output[batch_indices, flat_cumulative_valid_counts[flat_valid_mask]] = valid_embeddings
		label_lens, indx = label_lens.sort(dim=0, descending=True)
		valid_output = valid_output[indx]
		embedding_out = pack_padded_sequence(valid_output, lengths=label_lens.cpu(), batch_first=True) # unpad
		biGRU_out, _ = self.biGRU(embedding_out) # [batch_size, max_seq_length, 2*hidden_size1]
		biGRU_out, label_lens = pad_packed_sequence(biGRU_out, batch_first=True) # pad sequence to max length in batch
		GRU_out, _ = self.GRU(biGRU_out) # [batch_size, max_label_lens_in_this_batch, hidden_size2]
		# for punctuation prediction, concat with next token
		# Pad the tensor at the end of the T dimension to duplicate the last vector
		padded_tensor_punct = torch.nn.functional.pad(GRU_out, (0, 0, 0, 1), mode='replicate')
		# Concatenate the original tensor with the padded tensor, which includes the duplicated last vector
		concat_adjacent_punct = torch.cat((GRU_out, padded_tensor_punct[:, 1:, :]), dim=-1)
		punct_logits = self.decoder_punct(self.dropout2(concat_adjacent_punct))
		valid_ids = valid_ids[indx]
		# move all ones to the left
		valid_ids_sorted, _ = valid_ids.sort(dim=1, descending=True)
		valid_ids_sorted_sliced = valid_ids_sorted[:, :punct_logits.shape[1]]
		non_zero_mask = valid_ids_sorted_sliced != 0
		# exclude the first token, i.e. <bos>
		cumulative_non_zeros = valid_ids_sorted_sliced.cumsum(dim=1)
		exclude_first = cumulative_non_zeros != 1
		# exclude the last token, i.e. <eos>
		cumulative_non_zeros_flip = valid_ids_sorted_sliced.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
		exclude_last = cumulative_non_zeros_flip != 1
		final_mask = non_zero_mask & exclude_first & exclude_last
		active_punct_logits = punct_logits[final_mask] # (T', out_size_punct)
		return active_punct_logits, final_mask


