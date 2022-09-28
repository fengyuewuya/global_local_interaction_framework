#coding=utf-8
import sys
import os
import torch
import pickle
from torch_geometric.loader import DataLoader
from torch.nn.functional import mse_loss, l1_loss
from model.global_local_interaction_model import GlobalLocalInteractionModel

def get_corr(predict_Y, Y):
        predict_Y, Y = predict_Y.reshape(-1), Y.reshape(-1)
        predict_Y_mean, Y_mean = torch.mean(predict_Y), torch.mean(Y)
        corr = (torch.sum((predict_Y - predict_Y_mean) * (Y - Y_mean))) / (
                    torch.sqrt(torch.sum((predict_Y - predict_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
        return corr

class Evaluator:
    def __init__(self):
        pass

    def eval(self, input_dict):
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']
        result = {}
        MSE_loss = mse_loss(y_pred, y_true)
        MAE_loss = l1_loss(y_pred, y_true).cpu().item()
        coff = get_corr(y_pred, y_true).cpu().item()
        result["RMSE"] =  torch.sqrt(MSE_loss).cpu().item()
        result["MAE"] = MAE_loss
        result["R"] = coff
        return result

def test_model(model, data, device):
    vt_batch_size = 1
    test_loader = DataLoader(data, vt_batch_size, shuffle=False)
    evaluation = Evaluator()
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data.to(device)
        out = model(batch_data)
        preds = torch.cat([preds, out.detach_()], dim=0)
        targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)
    input_dict = {"y_true": targets, "y_pred": preds}
    result = evaluation.eval(input_dict)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pls input the trained_model_type like GLI-1-c, GLI-1-cg, GLI-1-cl, GLI-1-cgl!")
        exit()
    trained_model = sys.argv[1]
    if trained_model not in ["GLI-0-c", "GLI-0-cg", "GLI-0-cl", "GLI-0-cgl",
            "GLI-1-c", "GLI-1-cg", "GLI-1-cl", "GLI-1-cgl",
            "GLI-2-c", "GLI-2-cg", "GLI-2-cl", "GLI-2-cgl"]:
        print("Pls input the trained_model_type like GLI-1-c, GLI-1-cg, GLI-1-cl, GLI-1-cgl!")
        exit()
    # set chemical model type
    if trained_model.startswith("GLI-0"):
        chem_info_embedding_model_type = "gat_gcn"
    if trained_model.startswith("GLI-1"):
        chem_info_embedding_model_type = "gin"
    if trained_model.startswith("GLI-2"):
        chem_info_embedding_model_type = "gcn2"
    global_interaction = 0
    local_interaction = 0
    local_interaction_cutoff = 5
    if "g" in trained_model:
        global_interaction = 1
    if "l" in trained_model:
        local_interaction = 1
    model = GlobalLocalInteractionModel(
        chem_info_embedding_model_type=chem_info_embedding_model_type,
        global_interaction=global_interaction,
        local_interaction=local_interaction,
        local_interaction_cutoff=local_interaction_cutoff)
    # load model
    model_path = os.path.join("./trained_models", trained_model.strip() + ".pt")
    model_param = torch.load(model_path)["model_state_dict"]
    model.load_state_dict(model_param)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print("model_type, data_type, MAE, RMSE, R")
    # test in PDBbind v2016 core set
    data = pickle.load(open("./data/2016_core_data", "rb"))
    result = test_model(model, data, device)
    mae = result["MAE"]
    rmse = result["RMSE"]
    R = result["R"]
    print("%s, %s, %.3f, %.3f, %.3f" % (trained_model, "PDBbind v2016 core", mae, rmse, R))
    # test in CSAR_HiQ_data
    data = pickle.load(open("./data/CSAR_HiQ_data", "rb"))
    result = test_model(model, data, device)
    mae = result["MAE"]
    rmse = result["RMSE"]
    R = result["R"]
    print("%s, %s, %.3f, %.3f, %.3f" % (trained_model, "CSAR-HiQ", mae, rmse, R))

