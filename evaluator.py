from tensorflow.keras.callbacks import Callback
from utils.eval_utils import evaluate_predictions
from models.decoder import get_decoder_model


class EvaluatorCalback(Callback):
    def __init__(self, eval_data, eval_steps, labels, batch_size, prior_boxes, hyper_params) -> None:
        super().__init__()
        self.eval_data = eval_data
        self.eval_steps = eval_steps
        self.labels = labels
        self.batch_size = batch_size
        self.prior_boxes = prior_boxes
        self.hyper_params = hyper_params

    def on_epoch_end(self, epoch):
        ssd_decoder_model = get_decoder_model(self.model, self.prior_boxes, self.hyper_params)
        pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict(self.eval_data, steps=self.eval_steps, verbose=1)
        stats, mAP = evaluate_predictions(self.eval_data, pred_bboxes, pred_labels, pred_scores, self.labels, self.batch_size)