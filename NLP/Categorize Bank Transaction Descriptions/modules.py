from torch.optim import Adam
import torch

def train_on_epoch(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle = False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
      model = model.cuda()
      criterion = criterion.cuda()

    for epoch_num in range(epochs):

      total_acc_train = 0
      total_loss_train = 0
      train_loss_hist = []
      train_acc_hist = []

      optimizer.zero_grad()
      for train_input, train_label in tqdm(train_dataloader):

          train_label = train_label.to(device)
          mask = train_input['attention_mask'].to(device)
          input_id = train_input['input_ids'].squeeze(1).to(device)

          output = model(input_id, mask)

          batch_loss = criterion(output, train_label.long())
          total_loss_train += batch_loss.item()
          train_loss_hist.append(batch_loss.item())
          acc = (output.argmax(dim=1) == train_label).sum().item()
          total_acc_train += acc
          train_acc_hist.append(acc/len(train_label))
          model.zero_grad()
          batch_loss.backward()
          optimizer.step()

      total_acc_val = 0
      total_loss_val = 0
      val_loss_hist = []
      val_acc_hist = []

      with torch.no_grad():

        for val_input, val_label in val_dataloader:
          val_label = val_label.to(device)
          mask = val_input['attention_mask'].to(device)
          input_id = val_input['input_ids'].squeeze(1).to(device)

          output = model(input_id, mask)

          batch_loss = criterion(output, val_label.long())
          total_loss_val += batch_loss.item()
          val_loss_hist.append(batch_loss.item())
          acc = (output.argmax(dim=1) == val_label).sum().item()
          total_acc_val += acc
          val_acc_hist.append(acc/len(val_label))

      print(
          f'Fold: {fold} | Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
          | Train Accuracy: {total_acc_train / len(train_data): .3f} \
          | Val Loss: {total_loss_val / len(val_data): .3f} \
          | Val Accuracy: {total_acc_val / len(val_data): .3f}')
      
    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist
  
  
def evaluate(model, test_data, return_prob = True):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    preds = []
    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              if return_prob:
                output_prob = F.softmax(output, dim = 1)
                output_prob = output_prob.detach().cpu().numpy()
              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
              output = output.argmax(dim = 1).detach().cpu().numpy()
              preds = np.concatenate([preds, output], axis = 0)

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return preds, total_acc_test, output_prob
