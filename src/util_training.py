import os
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
import matplotlib
from src.utils import parseJson
import numpy as np
import torch



def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

def load_model(model, model_path, strict=False):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = False
def mkdir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path,exist_ok=True)
        print("The new directory {} is created!".format(path))

def draw_acc_loss_curve(resPath,savePath):

    matplotlib.use('Agg')

    data = parseJson(resPath)

    # train_res_file = parseJson(res_file_path)
    best_modelPath = data["best_model_path"]

    model_name = best_modelPath.split("/")[-1].split(".pkt")[0]
    imgPath = "{}/{}.png".format(savePath,model_name)

    train_accs = data['train_accs']
    train_pres = data['train_pres']
    train_recs = data['train_recs']
    train_f1s = data['train_f1s']
    train_loss = data['train_loss']
    eval_accs = data['eval_accs']
    eval_loss = data['eval_loss']
    eval_pres = data['eval_pres']
    eval_recs = data['eval_recs']
    eval_f1s = data['eval_f1s']

    test_metrics =  data['test_metrics'].split("\t")
    pre = test_metrics[-3]
    rec = test_metrics[-2]
    f1 = test_metrics[-1]

    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(32, 8))
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_accs, color='green', label='Training Acc')
    plt.plot(epochs, eval_accs, color='red', label='Validation Acc')
    # plt.title("ACC")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_loss, color='skyblue', label='Training Loss')
    plt.plot(epochs, eval_loss, color='blue', label='Validation Loss')
    # plt.title("LOSS")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()
    plt.title(model_name + "\nP:{} R:{} F1:{}".format(pre, rec, f1))

    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_pres, color='skyblue', label='Train precision')
    plt.plot(epochs, train_recs, color='blue', label='Train recall')
    plt.plot(epochs, train_f1s, color='red', label='Train f1')
    # plt.title("LOSS")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()


    plt.subplot(1, 4, 4)
    plt.plot(epochs, eval_pres, color='skyblue', label='Validation precision')
    plt.plot(epochs, eval_recs, color='blue', label='Validation recall')
    plt.plot(epochs, eval_f1s, color='red', label='Validation f1')
    # plt.title("LOSS")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()




    # plt.tight_layout()



    # plt.text(0.5, 1, test_metrics)
    # plt.text(-5, 60, 'Parabola $Y = x^2$', fontsize=22)
    plt.savefig(imgPath, dpi=120, bbox_inches='tight')  # dpi 代表像素
    # plt.show()
    plt.cla()







if __name__ == "__main__":
    setup_seed()