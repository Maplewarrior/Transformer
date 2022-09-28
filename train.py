import torch
import time
import numpy as np
from masking import get_source_mask, get_target_mask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, source_data, target_data, optim, scheduler, epochs, print_every=1e3, verbose=True):    
    model.to(device)
    model.train()
    start = time.time()
    tot_time = 0
    total_loss = 0
    loss_list = []

    source_all = source_data['input_ids']
    target_all = target_data['input_ids']
    
    num_sentences = len(source_all)
    # loop over epochs
    for epoch in range(epochs):
        print("\nEPOCH " +str(epoch+1)+" of "+str(epochs)+"\n")
        # loop over all sentences
        for i in range(num_sentences):            
            # unsqueeze to avoid dim mismatch between embedder and pe
            src = source_all[i].unsqueeze(0).to(device)
            trg = target_all[i].unsqueeze(0).to(device)
            # target input, remove last word
            trg_input = trg[:, :-1]
            
            # get targets
            y = trg[:, 1:].contiguous().view(-1)
            
            src_mask = get_source_mask(src.size(1), src).to(device)
            trg_mask = get_target_mask(trg_input.size(1), trg_input).to(device)
            
            preds, A = model.forward(src, trg_input, src_mask, trg_mask)

            optim.zero_grad()    
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y)
            loss.backward()
            optim.step()
            total_loss += loss.item()

            if verbose and i % print_every == 0:
                print("sentence:\t",i+1,"\ntime per batch:\t",np.round(time.time()-start, 2)/(i+1), "\nloss:\t", np.round(loss.item(),2), "\naverage loss:\t", np.round(total_loss,2)/(i+1)) 
        loss_list.append(total_loss/num_sentences)
        scheduler.step()
        end = time.time()
        elapsed = end-start
        tot_time += elapsed
        avg_time = tot_time/(epoch+1)
        est_remain = epochs*avg_time - tot_time
        print(f'Train Loss: {loss_list[epoch]:.4f}')
        print(f'Epoch took {elapsed:.1f}s')
        print(f'Estimated {(est_remain/60):.1f}m left')
