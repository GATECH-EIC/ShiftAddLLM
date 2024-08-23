import numpy as np
import torch
import matplotlib.pyplot as plt

MODEL_DIM = {
    "opt-125m": 768,
    "opt-350m": 1024,
    "opt-1.3b": 2048,
    "opt-2.7b": 2560,
    "opt-13b": 5120,
}

def get_data(file_name, model_name):
    quant_loss = {}
    with open(f'./sensitivity/{file_name}.txt', 'r') as f:
        for line in f:
            layer, loss = line.split(': ')
            if "fc" in layer:
                quant_loss["model.decoder.layers." + layer] = float(loss) / ((MODEL_DIM[model_name] ** 2) * 4)
            else:
                quant_loss["model.decoder.layers." + layer] = float(loss) / (MODEL_DIM[model_name] ** 2)
    return quant_loss


def get_score(analysis_result):
    layers = list(analysis_result.keys())
    weight_score = {}
    for each in layers:
        layer_range = analysis_result[each]["max"]["wh"] - analysis_result[each]["min"]["wh"]
        layer_std = analysis_result[each]["std"]["wh"]
        layer_mean = analysis_result[each]["mean"]["wh"]
        layer_norm = analysis_result[each]["norm"]["wh"]
        if "fc" in each:
            weight_score[each] = (layer_norm/2 * layer_std**2).item()
        else:
            weight_score[each] = (layer_norm * layer_std**2).item()
        # weight_score[each] = analysis_result[each]['norm']["wh"].item()
    return weight_score

def calculate_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def calculate_mean_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

def plot_fitted_function(weight_score, loss, title="fitted_function"):

    from scipy import stats
    TrueAcc = np.array(loss)
    SSDNA = np.array(weight_score)
    ssdnatau = stats.kendalltau(TrueAcc, SSDNA)
    pearsonr = stats.pearsonr(TrueAcc, SSDNA)
    spearmanr = stats.spearmanr(TrueAcc, SSDNA)
    print("Tau / Pearsonr / Spearmanr: {}\n{}\n{}\n".format(ssdnatau, pearsonr, spearmanr))

    # 创建一个新的图像
    fig, ax = plt.subplots(figsize=(2.5,2.5))

    # 绘制真实的loss值
    plt.scatter(weight_score, loss, c='#88c999', marker='o', alpha=0.8, s=8)

    # 计算拟合函数的值
    # x = np.linspace(min(weight_score), max(weight_score), 10000)
    # y = np.polyval([1,0], x)

    # 绘制拟合函数
    # plt.plot(x, y, label='Fitted function', color='red')
    # print(x, y, weight_score, loss)

    a, b = np.polyfit(weight_score, loss, deg=1)
    
    weight_score = np.array(weight_score)
    y_est = a * weight_score + b
    y_err = weight_score.std() * np.sqrt(1/len(weight_score) + (weight_score - weight_score.mean())**2 / np.sum((weight_score - weight_score.mean())**2))
    # plt.plot(weight_score, y_est, '-', color='g', alpha=0.5, linewidth=2)
    # plt.fill_between(weight_score, y_est - y_err, y_est + y_err, alpha=0.2, color='g')
    
    plt.tick_params(axis='both', labelsize=7)

    plt.xlabel('Rank w.r.t. Criteria', fontsize=9, fontweight='bold')
    plt.ylabel('Rank w.r.t. Reparam. Error', fontsize=9, fontweight='bold')
    plt.title(r'Kendall $\tau$ = 0.905', fontsize=9.5, fontweight='bold')

    ax.grid(axis='both')

    fig.subplots_adjust(hspace=0.)
    fig.tight_layout()

    plt.savefig(f"{title}.pdf", bbox_inches='tight')
    plt.close()


model_name = "opt-13b"
file = f"{model_name}-quant_loss"
quant_loss = get_data(file, model_name)
analysis_result = torch.load(f"./sensitivity/mixbit/{model_name}.pth")
weight_score = get_score(analysis_result)

sorted_quant_loss = sorted(quant_loss, key=quant_loss.get)
sorted_quant_loss = {key: rank for rank, key in enumerate(sorted_quant_loss, 1)}

sorted_weight_score = sorted(weight_score, key=weight_score.get)
sorted_weight_score = {key: rank for rank, key in enumerate(sorted_weight_score, 1)}

weight_score_rank = []
quant_loss_rank = []
for each_layer in sorted_weight_score.keys():
    weight_score_rank.append(sorted_weight_score[each_layer])
    quant_loss_rank.append(sorted_quant_loss[each_layer])

plot_fitted_function(weight_score_rank, quant_loss_rank, title="rank")
