import os
import logging
from collections import defaultdict
import numpy as np


class Trainer(object):
    def __init__(self):
        pass

    @staticmethod
    def _train(model, batch_generator, steps, summary_writer, save_summary_steps, log_every_n_batch=10):
        # TODO handle the summary_writer and save_summary_steps
        total_loss, n_batch_loss = 0.0, 0.0
        model.train()
        for i in range(steps):
            train_batch = batch_generator.next()

            # forward + backward + optimize
            model.zero_grad()
            loss = model(train_batch)
            loss.backward()
            model.update()

            if np.isnan(loss.item()):
                raise ValueError("NaN loss!")
            total_loss += loss.item()
            n_batch_loss += loss.item()
            if log_every_n_batch is not None and i > 0 and i % log_every_n_batch == 0:
                logging.info("- Average loss from batch {} to {} is {:05.3f}".format(
                             i - log_every_n_batch, i, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0.0

        logging.info("- Train mean loss: {:05.3f}".format(total_loss / steps))

    @staticmethod
    def _eval(model, batch_generator, steps, summary_writer=None):
        model.eval()
        final_output = defaultdict(list)
        for _ in range(steps):
            eval_batch = batch_generator.next()
            loss, output = model(eval_batch)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]

        # Get Eval Mean Loss
        logging.info("- Eval mean loss: " + loss.item())

        # Add summaries manually to writer at global_step_val
        if summary_writer is not None:
            # TODO
            pass
            # global_step_val = model.session.run(global_step)
            # for tag, val in metrics_val.items():
            #     summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            #     summary_writer.add_summary(summ, global_step_val)

        return final_output

    @staticmethod
    def _inference(model, batch_generator, steps):
        model.eval()
        final_output = defaultdict(list)
        for _ in range(steps):
            eval_batch = batch_generator.next()
            output = model(batch_generator)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]
        return final_output

    @staticmethod
    def train_and_evaluate(model, train_batch_generator, eval_batch_generator, evaluator, epochs=1, episodes=1,
                            save_dir=None, summary_dir=None, save_summary_steps=10):
        # TODO use tensorboardX
        train_summary = None
        eval_summary = None
        # train_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'train_summaries')) if summary_dir else None
        # eval_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'eval_summaries')) if summary_dir else None

        best_eval_score = 0.0
        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))
            train_num_steps = (train_batch_generator.get_dataset_size() +
                               train_batch_generator.get_batch_size() - 1) // train_batch_generator.get_batch_size()

            # one epoch consists of several episodes
            assert isinstance(episodes, int)
            num_steps_per_episode = (train_num_steps + episodes - 1) // episodes
            for episode in range(episodes):
                logging.info("episode {}/{}".format(episode + 1, episodes))
                current_step_num = min(num_steps_per_episode, train_num_steps - episode * num_steps_per_episode)
                episode_id = epoch * episodes + episode + 1
                Trainer._train(model, train_batch_generator, current_step_num, train_summary, save_summary_steps)

                if model.ema_decay>0:
                    # TODO how to do it
                    pass

                # Save weights
                if save_dir is not None:
                    last_save_path = os.path.join(save_dir, 'last_weights', 'after-episode')
                    model.save(last_save_path, global_step=episode_id)

                # Evaluate for one episode on dev set
                eval_dataset = eval_batch_generator.get_dataset()

                eval_num_steps = (eval_batch_generator.get_dataset_size() +
                                  eval_batch_generator.get_batch_size() - 1) // eval_batch_generator.get_batch_size()
                output = Trainer._eval(model, eval_batch_generator, eval_num_steps, eval_summary)
                score = evaluator.get_score(model.get_best_answer(output, eval_dataset))

                metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
                logging.info("- Eval metrics: " + metrics_string)

                if model.ema_decay>0:
                    # TODO how to do it
                    pass

                # Save best weights
                eval_score = score[evaluator.get_monitor()]
                if eval_score > best_eval_score:
                    logging.info("- epoch %d episode %d: Found new best score: %f" % (epoch + 1, episode + 1, eval_score))
                    best_eval_score = eval_score
                    # Save best weights
                    if save_dir is not None:
                        best_save_path = os.path.join(save_dir, 'best_weights', 'after-episode')
                        # TODO the best save path need always only one model, need to be improved
                        for file in os.listdir(best_save_path):
                            os.remove(os.path.join(best_save_path, file))
                        model.save(best_save_path, global_step=episode_id)
                        logging.info("- Found new best model, saving in {}".format(best_save_path))

    @staticmethod
    def evaluate(model, batch_generator, evaluator):
        eval_dataset = batch_generator.get_dataset()

        eval_num_steps = (eval_dataset.get_dataset_size() +
                          batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer._eval(model, batch_generator, eval_num_steps, None)
        pred_answer = model.get_best_answer(output, eval_dataset)
        score = evaluator.get_score(pred_answer)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
        logging.info("- Eval metrics: " + metrics_string)

    @staticmethod
    def inference(model, batch_generator):
        instances = batch_generator.get_dataset()
        eval_num_steps = (len(instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer._inference(model, batch_generator, eval_num_steps)
        pred_answers = model.get_best_answer(output, instances)
        return pred_answers
