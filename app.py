from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
import torch
import personCreator

tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
model = AutoModelForCausalLM.from_pretrained("af1tang/personaGPT")
 
app = Flask(__name__)

dialog_hx = []

if torch.cuda.is_available():
    model = model.cuda()
## utility functions ##
flatten = lambda l: [item for sublist in l for item in sublist]

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()

def generate_next(bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                  max_length=1000, pad_token=tokenizer.eos_token_id):
    full_msg = model.generate(bot_input_ids, do_sample=True,
                                              top_k=top_k, top_p=top_p, 
                                              max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    msg = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
    return msg


@app.route("/")
def home():
    
    return render_template('index.html')


@app.route("/get")
def get_bot_response():

    userText = request.args.get('msg')
    personas = []

    response = "My name is John, I am 75 years old and I feel quite old and ugly."+ tokenizer.eos_token
    personas.append(response)
    response = "I am a farmer in a small village."+ tokenizer.eos_token
    personas.append(response)
    response = "I have two daughters but they don't love me. I don't see them."+ tokenizer.eos_token
    personas.append(response)
    response = "I hate talking or interacting with people"+ tokenizer.eos_token
    personas.append(response)
    response = "The only thing I like is fishing."+ tokenizer.eos_token
    personas.append(response)
    response = "I drink wine all day and do nothing."+ tokenizer.eos_token
    personas.append(response)

    personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))
    
    for step in range(1):
        """new_user_input_ids = tokenizer.encode(userText + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000,pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7,temperature = 0.8)
        return str(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))"""

        # encode the user input
        user_inp = tokenizer.encode(userText + tokenizer.eos_token)
        # append to the chat history
        dialog_hx.append(user_inp)
        
        # generated a response while limiting the total chat history to 1000 tokens, 
        bot_input_ids = to_var([personas + flatten(dialog_hx)]).long()
        msg = generate_next(bot_input_ids)
        dialog_hx.append(msg)
        return str(tokenizer.decode(msg, skip_special_tokens=True))


 
 
if __name__ == "__main__":
        app.run(debug=False, host='0.0.0.0', port=8080)