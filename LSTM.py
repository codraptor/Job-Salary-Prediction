import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, embedding):

        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        if(embedding):
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim).cuda()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            batch_first=True).cuda()
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size).cuda()
        self.sigm = nn.Sigmoid()        

    def forward(self, x, hidden):
  
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim).cuda()
        
        # fully-connected layer
        out = self.fc(lstm_out)

        # sigmoid function
        sigmoid_out = self.sigm(out)
        
        # reshape to be batch_size first
        sigmoid_out = sigmoid_out.view(batch_size, -1)
        sigmoid_out = sigmoid_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sigmoid_out, hidden
    
    
    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
        