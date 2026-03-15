import torch
from tqdm.auto import tqdm

def train_step(model,
               train_dataloader,
               loss_fn,
               optimizer):
    train_loss = 0
    train_acc = 0

    for batch,(X,y) in enumerate(train_dataloader):
        model.train()
        y_logits = model(X)
        y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
        acc = torch.eq(y_pred,y).sum().item() / len(y)
        loss = loss_fn(y_logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    return train_loss, train_acc

def test_step(
        model,
        test_dataloader,
        loss_fn
        ):
    test_loss = 0
    test_acc = 0

    for X,y in test_dataloader:
        model.eval()
        with torch.inference_mode():
            y_logits = model(X)
            y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1) 
            acc = torch.eq(y_pred,y).sum().item() / len(y)
            loss = loss_fn(y_logits,y)
            test_loss += loss.item()
            test_acc += acc
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    return test_loss, test_acc

def train(
        model,
        train_dataloader,
        test_dataloader,
        loss_fn,
        optimizer,
        epochs ):
    results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
    }
    for epoch in tqdm(range(epochs)):

        train_loss , train_acc = train_step(
            model,
            train_dataloader,
            loss_fn,
            optimizer
        )

        test_loss , test_acc = test_step(
            model,
            test_dataloader,
            loss_fn
        )

        print(
            f"epoch : {epoch + 1} | "
            f"train loss : {train_loss} | "
            f"train acc : {train_acc * 100} % | "
            f"test loss : {test_loss} | "
            f"test acc : {test_acc * 100} %  "
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results

    


        