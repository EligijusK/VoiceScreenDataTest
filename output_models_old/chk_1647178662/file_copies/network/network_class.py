from network.state import set_state
from .base_network import BaseNetwork
from .models.model_index import ModelIndex


class NetworkClass(BaseNetwork):
    @property
    def ModelKey(self) -> str:
        return "ModelClass"

    @staticmethod
    def globals():
        return globals()

    def initModel(self):
        return

    def _write_log(*, model, data_testing, args, last_state, epoch, summary_training, summary_testing, epoch_results):
        train_loss, train_acc = epoch_results["loss"], epoch_results["accuracy"]

        train_loss_total = train_loss["loss"] if "loss" in train_loss else None
        train_loss_class = train_loss["class"] if "class" in train_loss else None

        train_acc_class = train_acc["class"] if "class" in train_acc else None

        last_state.epoch = epoch + 1
        set_state(last_state.train, train_loss_total)

        if epoch >= args.testing_set:
            test_result = model.evaluate(data_testing)

            test_loss, test_acc = test_result["loss"], test_result["accuracy"]

            test_loss_total = test_loss["loss"] if "loss" in test_loss else None
            test_loss_class = test_loss["class"] if "class" in test_loss else None

            test_acc_class = test_acc["class"] if "class" in test_acc else None

            set_state(last_state.test, test_loss_total)

        with summary_training as writer:
            if train_loss_total is not None:
                writer.add_scalar("loss/total", train_loss_total, epoch)
                writer.add_scalar("loss/class", train_loss_class, epoch)

                writer.add_scalar("accuracy/class", train_acc_class, epoch)

        if epoch >= args.testing_set:
            with summary_testing as writer:
                if test_loss_total is not None:
                    writer.add_scalar("loss/total", test_loss_total, epoch)
                    writer.add_scalar("loss/class", test_loss_class, epoch)

                    writer.add_scalar("accuracy/class", test_acc_class, epoch)

        return train_loss_total, test_loss_total