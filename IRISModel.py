# ------------------------------------------------------------
# Cria o módulo IRISModel, conforme arquivo no tutorial:
# https://github.com/FutureLab-DCC/flautim_tutoriais/tree/main/TUTORIAL_1
#
# Nesta seção, o módulo IRISModel.py é armazenado como um arquivo local.
# O objetivo dessa abordagem é simular uma estrutura semelhante à adotada
# na plataforma Flautim, na qual os scripts responsáveis por definir a
# arquitetura do modelo são mantidos de forma modular e independente. Dessa
# forma, o  arquivo pode posteriormente ser submetido (upload) na área de Modelos
# da plataforma, permitindo sua reutilização, versionamento e integração
# com diferentes experimentos.
# ------------------------------------------------------------
# ------------------------------------------------------------
from flautim.pytorch.Model import Model
import torch

class IRISModel(Model):
    def __init__(self, context, num_classes = 3, **kwargs):
        super(IRISModel, self).__init__(context, name = "IRIS-NN", **kwargs)

        # Rede neural com 4 entradas e 3 saídas
        self.c1 = torch.nn.Linear(4, 10)
        self.c2 = torch.nn.Linear(10, num_classes)


    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        return x