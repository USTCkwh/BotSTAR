import copy
from tqdm import tqdm
import torch
from models.adapt_model import Adaptation
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
import torch.nn.functional as F

def null_metrics():
    return {
        'accuracy': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
    }


def compute_metrics(y_true, y_output):
    metrics = null_metrics()
    if torch.any(torch.isnan(y_output)):
        metrics['msg'] = 'NaN in y_output'
        print('error!')
    else:
        y_pred = F.softmax(y_output, dim=-1)
        y_true = y_true.to('cpu').detach().numpy()
        y_pred_label = torch.argmax(y_pred, dim=-1).to('cpu').detach().numpy()
        y_pred_score = y_pred[:, -1].to('cpu').detach().numpy()
        precision, recall, _thresholds = precision_recall_curve(y_true, y_pred_score)
        metrics['msg'] = 'success'
        metrics['accuracy'] = round(accuracy_score(y_true, y_pred_label), 5)
        metrics['f1'] = round(f1_score(y_true, y_pred_label), 5)
        metrics['precision'] = round(precision_score(y_true, y_pred_label), 5)
        metrics['recall'] = round(recall_score(y_true, y_pred_label), 5)
    return metrics


def is_better(now, pre):
    if now['accuracy'] >= pre['accuracy']:
        return True
    else:
        return False



def node_classification_evaluation(pre_model, graph, num_hidden, lr_f, weight_decay_f, max_epoch_f, finetune_drop, device, dataset, basevector_num, bvlen_factor=1, adapt_align_loss_rate=1.0, edge_add_ratio=0.5):
    gnn_encoder = copy.deepcopy(pre_model.gnn_encoder)
    model = Adaptation(gnn_encoder, num_hidden, finetune_drop, dataset, basevector_num, bvlen_factor=bvlen_factor, edge_add_ratio=edge_add_ratio)
    model.to(device)
    optimizer_f = optim.Adam(model.parameters(), lr=lr_f, weight_decay=weight_decay_f)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_f, T_max=300, eta_min=0)
    graph = graph.to(device)
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    labels = graph.y
    best_val_metric = null_metrics()
    best_model = None
    epoch_iter = tqdm(range(max_epoch_f))
    best_val_epoch = 0
    best_test_metrics = null_metrics()
    best_test_epoch = 0
    align_weight = adapt_align_loss_rate
    for epoch in epoch_iter:
        model.train()
        out, align_loss = model(graph)
        train_acc = compute_metrics(labels[train_mask], out[train_mask])['accuracy']
        loss = criterion(out[train_mask], labels[train_mask]) + align_weight * align_loss
        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()
        scheduler.step()
        with torch.no_grad():
            model.eval()
            pred, _ = model(graph)
            val_metrics = compute_metrics(labels[val_mask], pred[val_mask])
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1']
            val_loss = criterion(pred[val_mask], labels[val_mask])

            test_metrics = compute_metrics(labels[test_mask], pred[test_mask])
            test_acc = test_metrics['accuracy']
            test_f1 = test_metrics['f1']
            
            # if (epoch + 1) % 10 == 0:
            #     pre_model.gnn_encoder.load_state_dict(model.gnn_encoder.state_dict())
            #     mask_recon_loss, _ = pre_model.mask_attr_prediction(graph.x, graph.edge_index, graph.edge_type, epoch)
            #     mask_recon_losses[epoch+1] = mask_recon_loss.item()
        if is_better(val_metrics, best_val_metric):
            best_val_metric = val_metrics
            best_model = copy.deepcopy(model)
            best_val_epoch = epoch
        if is_better(test_metrics, best_test_metrics):
            best_test_metrics = test_metrics
            best_test_epoch = epoch
        epoch_iter.set_description(
            f'# Epoch: {epoch}, train_loss:{loss.item(): .4f}, train_acc:{train_acc},'
            f' val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, val_f1:{val_f1},')
    with torch.no_grad():
        model.eval()
        pred, _ = model(graph)
        final_metrics = compute_metrics(labels[test_mask], pred[test_mask])
    with torch.no_grad():
        best_model.eval()
        pred, _ = best_model(graph)
        estp_metrics = compute_metrics(labels[test_mask], pred[test_mask])
    print(f"--- Final-epoch-TestAcc: {final_metrics['accuracy']:.4f}, "
          f"best-val-epoch-TestAcc: {estp_metrics['accuracy']:.4f}, "
          f"best-val-epoch: {best_val_epoch}, ")
    final_model = model
    estp_model = best_model
    return final_metrics, estp_metrics, final_model, estp_model


