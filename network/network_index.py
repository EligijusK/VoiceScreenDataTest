from network.state import set_state
from .base_network import BaseNetwork
from .models.model_index import ModelIndex

class NetworkIndex(BaseNetwork):
    @property
    def ModelKey(self) -> str:
        return "ModelIndex"

    @staticmethod
    def globals():
        return globals()

    def initModel(self):
        return self.ModelConstructor()

    def _write_log(*, model, data_testing, args, last_state, epoch, summary_training, summary_testing, epoch_results):
        train_loss = epoch_results["loss"]

        train_loss_sum = train_loss["loss"] if "loss" in train_loss else None
        train_loss_shi = train_loss["shi"] if "shi" in train_loss else None
        train_loss_total = train_loss["total"] if "total" in train_loss else None

        last_state.epoch = epoch + 1
        set_state(last_state.train, train_loss_sum)

        if epoch >= args.testing_set:
            test_result = model.evaluate(data_testing)

            test_loss = test_result["loss"]

            test_loss_sum = test_loss["loss"] if "loss" in test_loss else None
            test_loss_shi = test_loss["shi"] if "shi" in test_loss else None
            test_loss_total = test_loss["total"] if "total" in test_loss else None

            set_state(last_state.test, test_loss_sum)

        with summary_training as writer:
            if train_loss_sum is not None:
                writer.add_scalar("loss/sum", train_loss_sum, epoch)
                if train_loss_shi is not None: writer.add_scalar("loss/shi", train_loss_shi, epoch)
                if train_loss_total is not None: writer.add_scalar("loss/total", train_loss_total, epoch)

        if epoch >= args.testing_set:
            with summary_testing as writer:
                if test_loss_sum is not None:
                    writer.add_scalar("loss/sum", test_loss_sum, epoch)
                    if test_loss_shi is not None: writer.add_scalar("loss/shi", test_loss_shi, epoch)
                    if test_loss_total is not None: writer.add_scalar("loss/total", test_loss_total, epoch)

        return test_loss_sum, test_loss_sum