import time
import torch
from torch.utils import secure_mkldnn as mkldnn_utils
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
model.eval()
print(model)

print("=========================================")
print("call original pytorch backend")
k = input("press any key to continue:")

dummy_input = torch.ones(1, 3, 224, 224)

start_time_thnn = time.time()
with torch.backends.mkldnn.flags(enabled=False):
    print(model(dummy_input))
end_time_thnn =  time.time()
print("--- %s seconds ---" % (end_time_thnn - start_time_thnn))


print("=========================================")
print("call mkldnn in sgx")
k = input("press any key to continue:")

model = mkldnn_utils.to_secure_mkldnn(model)
start_time_mkldnn = time.time()
print(model(dummy_input))
end_time_mkldnn =  time.time()
print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))
torch.save(model.state_dict(), "./secure_resnet.pt")
