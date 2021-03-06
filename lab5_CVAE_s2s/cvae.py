import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
# Hyper Parameters
hidden_size = 256
conditional_size = 8
# The number of vocabulary
vocab_size = 28
latent_size = 32


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).view(1, 1, -1) # view(1,1,-1) due to input of rnn must be (seq_len,batch,vec_dim)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return output, hidden, cell

    def initHidden(self, size):
        return torch.zeros(1, 1, size, device=device)

    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.out(output[0])
        return output, hidden, cell

    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def reparameterize(mean, log_var):
    """
    reparameterization trick
    """
    eps = torch.randn_like(log_var, device=device)
    return eps * torch.exp(0.5 * log_var) + mean

class CVAE(nn.Module):
    def __init__(self, max_length):
        super(CVAE, self).__init__()
        self.encoder = EncoderRNN(vocab_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, vocab_size)
        self.max_length = max_length
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        self.conditional2embbed = nn.Embedding(4, conditional_size)
        self.latent2embedd = nn.Linear(latent_size + conditional_size, hidden_size)
    
    def forward(self, input_tensor, tense_tensor, teacher_forcing_ratio, inference=False):
        if inference:
            output_length = self.max_length
        else:
            output_length = len(input_tensor)

        init_hidden = self.encoder.initHidden(hidden_size - conditional_size)
        c = self.conditional2embbed(tense_tensor)
        encoder_hidden = torch.cat((init_hidden, c),dim=-1)
        encoder_cell = self.encoder.initCell()
        # encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)
        
        #----------sequence to sequence part for encoder----------#
        for i in range(len(input_tensor)):
            _, encoder_hidden, encoder_cell = self.encoder(input_tensor[i], encoder_hidden, encoder_cell)
        
        # sample point
        mean = self.hidden2mean(encoder_hidden)
        log_var = self.hidden2logvar(encoder_hidden)
        latent = reparameterize(mean, log_var)

        #----------sequence to sequence part for encoder----------#
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = torch.cat((latent, c), dim=-1)

        # decoder_hidden = decoder_hidden.long()
        decoder_hidden = self.latent2embedd(decoder_hidden)
        decoder_cell = self.decoder.initCell()
        predict_distribution = torch.zeros(output_length, vocab_size, device=device)
        output=[]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        for i in range(output_length):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            predict_distribution[i] = decoder_output
            predict_class = torch.argmax(decoder_output)
            output.append(predict_class)
            if use_teacher_forcing:
                decoder_input = input_tensor[i]
            else:
                if predict_class.item() == EOS_token:
                    break
                decoder_input = predict_class

        return output, predict_distribution, mean, log_var
    
    def generate_words(self, latent, tense_list):
        word_list = []
        for tense in tense_list:  # 4 tense
            tense_tensor = torch.tensor(tense, device=device).view(-1,1)
            c = self.conditional2embbed(tense_tensor)
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = torch.cat((latent, c), dim=-1)

            # decoder_hidden = decoder_hidden.long()
            decoder_hidden = self.latent2embedd(decoder_hidden)
            decoder_cell = self.decoder.initCell()
            predict_word=[]

            for i in range(self.max_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                predict_class = torch.argmax(decoder_output)
                predict_word.append(predict_class)
                if predict_class.item() == EOS_token:
                    break
                decoder_input = predict_class
            
            word_list.append(predict_word)

        return word_list

