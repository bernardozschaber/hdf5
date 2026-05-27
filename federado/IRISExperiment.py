# ------------------------------------------------------------
# Cria o módulo IRISExperiment, conforme arquivo no tutorial:
# https://github.com/FutureLab-DCC/flautim_tutoriais/tree/main/TUTORIAL_1
#
# Obs.: Foi adicionado comandos para escrever no arquivo HDF5
#       informações geradas a cada execução do validation_loop
# ------------------------------------------------------------
from flautim.pytorch.centralized.Experiment import Experiment
import flautim as fl
import numpy as np
import torch
import time

import io
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class IRISExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(IRISExperiment, self).__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.epochs = kwargs.get('epochs', 30)


    def training_loop(self, data_loader):
        self.model.train()
        error_loss = 0.0
        yhat, y_real = [], []

        for X, y in data_loader:
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            error_loss += loss.cpu().item()
            _, predicted = torch.max(outputs.data, 1)
            yhat.append(predicted.detach().cpu())
            y_real.append(y.detach().cpu())

        accuracy = self.metrics.ACCURACY(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        accuracy_2 = self.metrics.ACCURACY_2(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        error_loss = error_loss / len(data_loader)

        return float(error_loss), {'ACCURACY': accuracy, 'ACCURACY_2': accuracy_2}

    def validation_loop(self, data_loader):
        error_loss = 0.0
        yhat, y_real = [], []
        self.model.eval()

        with torch.no_grad():
            for X, y in data_loader:
                outputs = self.model(X)
                error_loss += self.criterion(outputs, y).item()
                _, predicted = torch.max(outputs.data, 1)
                yhat.append(predicted.detach().cpu())
                y_real.append(y.detach().cpu())

        accuracy = self.metrics.ACCURACY(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        accuracy_2 = self.metrics.ACCURACY_2(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        error_loss = error_loss / len(data_loader)

        # ====================================================================
        # Exemplo utilizando a funcionalidade de armazenamento de outputs em
        # arquivos HDF5. Este trecho consiste em uma extensão do Tutorial 1
        # original, que aborda o uso do dataset IRIS, incorporando agora
        # mecanismos adicionais para persistência dos resultados gerados
        # durante a execução. A proposta é demonstrar, de forma prática, como
        # integrar o fluxo tradicional de manipulação do dataset IRIS com o
        # armazenamento estruturado em HDF5, permitindo o registro organizado
        # de saídas, métricas e artefatos produzidos ao longo do experimento.
        # ====================================================================
        y_pred = torch.cat(yhat).numpy()
        y_true = torch.cat(y_real).numpy()

        #CONFUSION MATRIX (sem sklearn)
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
                cm[int(t), int(p)] += 1

		    #GERAR IMAGEM (PNG em memória)
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        epoch = int(getattr(self, "epochs", -1))

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm)  # sem escolher cor fixa (fica default do matplotlib)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
                f"Confusion Matrix | epoch={epoch}\n"
                f"{now_str}\n"
                f"ACC={float(accuracy):.4f} | ACC2={float(accuracy_2):.4f}"
        )

		    # escreve os números em cima das células (fica bem legível)
        for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        fig.colorbar(im, ax=ax)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.read()

		    # SALVAR OUTPUT (imagem)
        img_name = f"confusion_epoch_{epoch}_{now.strftime('%Y%m%d_%H%M%S')}.png"
        img_ref = fl.output_image( png_bytes, name=img_name, meta={ "epoch": int(epoch), "created_at": now.isoformat(),  "ACCURACY_2": float(accuracy_2) } )


		    # SALVAR OUTPUT (ARRAY) com o y_pred e y_true
        pairs = np.column_stack((y_true.astype(np.int64), y_pred.astype(np.int64)))
        pairs_ref = fl.output_array(
            pairs,
            name=f"y_true_y_pred_epoch_{epoch}.npy",
            meta={
                "epoch": int(epoch),
                "type": "y_true_y_pred_pairs",
                "columns": ["y_true", "y_pred"],
                "n": int(pairs.shape[0]),
            },
        )

		    # SALVAR OUTPUT (JSON) com métricas + matriz
        json_name = f"validation_epoch_{epoch}_{now.strftime('%Y%m%d_%H%M%S')}.json"
        json_ref = fl.output_json(
          {
            "epoch": int(epoch),
            "created_at": now.isoformat(),
            "loss": float(error_loss),
            "metrics": {
              "ACCURACY": float(accuracy),
              "ACCURACY_2": float(accuracy_2),
            },
                    "predictions_pairs_ref": pairs_ref,
            "confusion_matrix": cm.tolist(),
            "confusion_image_ref": img_ref,
          },
          name=json_name,
          meta={
            "epoch": int(epoch),
            "created_at": now.isoformat(),
            "loss": float(error_loss),
            "ACCURACY": float(accuracy),
            "ACCURACY_2": float(accuracy_2),
          },
        )

		    # registrar um log apontando pros outputs
        fl.log(
          f"Validation outputs saved (epoch={epoch})",
          details=f"img_ref={img_ref} json_ref={json_ref}",
          object="validation",
        )
        # ====================================================================

        return float(error_loss), {'ACCURACY': accuracy, 'ACCURACY_2': accuracy_2}