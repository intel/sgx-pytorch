# https://pytorch.org/hub/pytorch_vision_resnet
import torch
import urllib
import urllib.request
from PIL import Image
from torchvision import transforms
from torch.utils import mkldnn
from torch.utils import secure_mkldnn
import time

#model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
model.eval()

model2 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
model2.eval()

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#try: urllib.URLopener().retrieve(url, filename)
#except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

model2 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
model2.eval()
input_batch2 = input_batch


###############

start_time_mkldnn = time.time()
#with torch.backends.mkldnn.flags(enabled=False):
#    output = model(input_batch)

output = model(input_batch)

end_time_mkldnn =  time.time()
#print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))

print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
print(probabilities.sum())

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))


###################################################


input("press any key to start secure op...")


model = secure_mkldnn.to_secure_mkldnn(model, model_id=(0).to_bytes(4, byteorder = 'little'))
start_time_mkldnn = time.time()
output = model(input_batch)
end_time_mkldnn =  time.time()
#print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))

print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
print(probabilities.sum())

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))

###############################################

input("2 press any key to start secure op...")


model = secure_mkldnn.to_secure_mkldnn(model)
start_time_mkldnn = time.time()
output = model(input_batch)
end_time_mkldnn =  time.time()
#print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))

print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
print(probabilities.sum())

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))


#############################################################


input("press any key to start mkldnn...")

model2 = mkldnn.to_mkldnn(model2)
input_batch2 = input_batch2.to_mkldnn()

start_time_mkldnn = time.time()
output = model2(input_batch2)
end_time_mkldnn =  time.time()
print("--- %s seconds ---" % (end_time_mkldnn - start_time_mkldnn))

