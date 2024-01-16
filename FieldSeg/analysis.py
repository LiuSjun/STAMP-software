import os

from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt

TS_ROOT = r"E:\Newsegdataset\xizang\tensorboard"


def ts_reader(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    val_psnr = ea.scalars.Items(ea.scalars.Keys()[0])
    time = [i.wall_time for i in val_psnr]
    e = [i.step for i in val_psnr]
    metric = [i.value for i in val_psnr]
    return time, e, metric


def ts_file_reader(root, metric, dts):
    '''
    通过每一次训练的根目录文件，获得train, val and test的metric参数
    '''
    tra_even_root = os.path.join(root, "{}\{}".format(dts, metric))
    tra_even_path = os.listdir(tra_even_root)[0]
    tra_even_path = os.path.join(tra_even_root, tra_even_path)

    time, e, metric = ts_reader(tra_even_path)
    return metric


def comp_f1_ac(model_a_root, model_b_root, model_a_name, model_b_name):
    path = os.path.join(TS_ROOT, model_a_root)
    model_a_f1 = ts_file_reader(path, 'F1', 'test')
    model_a_ac = ts_file_reader(path, 'AC', 'test')

    path = os.path.join(TS_ROOT, model_b_root)
    model_b_f1 = ts_file_reader(path, 'F1', 'test')
    model_b_ac = ts_file_reader(path, 'AC', 'test')

    plt.figure()
    plt.subplot(121)
    plt.title("F1")
    plt.plot(model_a_f1, label=model_a_name)
    plt.plot(model_b_f1, label=model_b_name)

    plt.subplot(122)
    plt.title("AC")
    plt.plot(model_a_ac, label=model_a_name)
    plt.plot(model_b_ac, label=model_b_name)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Reference
    metrics_names = ['F1', 'AC', 'MSE', 'BCE', 'Tani']
    dataset = ["tra", "test", "val"]

    model_a_file = 'base'   #'reuet_bou_grad'
    model_b_file = 'reuet_bou'  #'reuet_bou'

    comp_f1_ac(model_a_file, model_b_file, 'Boundary', 'Boundary with gradient')