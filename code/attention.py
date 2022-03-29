import torch
import torch.nn.functional as f

class Attention_Layer(torch.nn.Module):
	def __init__(self, vector_count, embedding_dim):
		super(Attention_Layer, self).__init__()
		self.embedding_dim = embedding_dim
		self.vector_count = vector_count
		self.keys = torch.nn.Linear(embedding_dim, embedding_dim)
		self.queries = torch.nn.Linear(embedding_dim, embedding_dim)
		self.values = torch.nn.Linear(embedding_dim, embedding_dim)

	def forward(self, x, return_weights=False):
		keys_x = self.keys(x)
		queries_x = self.queries(x)
		values_x = self.values(x)
		scores = torch.bmm(queries_x, keys_x.transpose(1, 2))
		scale = queries_x.size(-1) ** 0.5
		softmax = f.softmax(scores / scale, dim=-1)
		y = softmax.bmm(values_x)
		if return_weights:
			return y, weights
		else:
			return y

if __name__ == '__main__':
	vector_count = 10
	embedding_dim = 16
	x = torch.randn((64, vector_count, embedding_dim))
	al = Attention_Layer(vector_count=vector_count, embedding_dim=embedding_dim)
	y = al(x)
	print(y.shape)