import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from google_model import GoogLeNet

def test_data_process():
    #数据下载
    test_data = FashionMNIST(root="../data",
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                              download=True)

    #加载训练数据到数据容器
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_dataloader

def test_model_process(model,test_dataloader):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    test_crrects = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)

            pre_lab = torch.argmax(output, dim=1)

            test_crrects += torch.sum(pre_lab==test_data_y)

            test_num += test_data_x.size(0)

    test_acc = test_crrects.double().item() / test_num
    print("-"*20)
    print("测试是的准确率为:",test_acc)
    print("-" * 20)


if __name__ == "__main__":
    lenet_test = GoogLeNet()
    lenet_test.load_state_dict(torch.load("best_model.pth"))
    test_dataloader = test_data_process()
    test_model_process(lenet_test, test_dataloader)
    crrect_pred=0
    total_num=0
    device = torch.device("cuda")
    lenet_test.to(device)
    catgory = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            lenet_test.eval()
            output = lenet_test(b_x)
            result = torch.argmax(output,dim=1).item()
            print("预测值为：",catgory[result],"真实值为：",catgory[b_y.item()])








