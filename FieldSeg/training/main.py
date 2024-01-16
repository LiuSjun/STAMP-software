from training.main_bou import tranining as tr_bou
from training.main_bou_grad import tranining as tr_bou_grad

###### IMPORT Training Data
from dataset.dataset_INT_aug import DatasetGF2
from dataset.datasets.dataset_ah_bou import DatasetAH
from dataset.datasets.dataset_cdbou import DatasetCD
from dataset.datasets.dataset_gsbou import DatasetGS
from dataset.datasets.dataset_hljbou import DatasetHLJ
from dataset.datasets.dataset_msbou import DatasetMS
from dataset.datasets.dataset_zjbou import DatasetZJ
from dataset.datasets.dataset_zzbou import DatasetZZ
from dataset.datasets.dataset_testbou import DatasetT
from dataset.datasets.dataset_fnbou import DatasetFN

datasets = [DatasetAH(), DatasetCD(), DatasetGS(), DatasetFN(), DatasetMS(), DatasetZJ(), DatasetZZ(), DatasetT()]
dataset = DatasetGF2(
    datasets
)

if __name__ == "__main__":
    dataset_tst = DatasetT()
    bestpth = ''
    dataset_tra, dataset_val = dataset.dataset_split(0.9)
    dataset_val.training = False  # Validation Set上不执行数据增强
    tr_bou(dataset_tra, dataset_val, dataset_tst, bestpth)
    tr_bou_grad(dataset_tra, dataset_val, dataset_tst, bestpth)