import time
import torch
from torch.utils import secure_mkldnn as mkldnn_utils
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
model.eval()
dummy_input = torch.ones(1, 3, 224, 224)

model = mkldnn_utils.to_secure_mkldnn(model)
model.load_state_dict(torch.load('./secure_resnet.pt'))
start_time_mkldnn = time.time()
print(model(dummy_input))
end_time_mkldnn =  time.time()
print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))


