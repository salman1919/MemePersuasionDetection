import torchvision
from transformers import RobertaModel, RobertaTokenizer


def get_pre_trained_text_model(device="cpu"):
    model_name = 'roberta-base'  # or any other pre-trained model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    roberta_model = RobertaModel.from_pretrained(model_name)
    roberta_model.to(device)

    return roberta_model, tokenizer


def get_pre_trained_image_model(device="cpu"):
    resnet_model = torchvision.models.resnet152(weights='DEFAULT')
    for param in resnet_model.parameters():
        param.requires_grad = False
    resnet_model.to(device)

    return resnet_model
