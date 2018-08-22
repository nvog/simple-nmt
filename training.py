import math
import os
import random
import time

from batch import create_batches


class BasicTrainingProcedure:
    def __init__(self, model, trainer):
        self.trainer = trainer
        self.model = model

    def train(self, num_epochs, train, dev, train_batch_size, dev_batch_size, max_sent_size):
        # training
        training_timestamp = int(time.time())
        model_path = "models/{}".format(training_timestamp)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print("Creating batches")
        train_batches = create_batches(train, train_batch_size, max_sent_size)
        dev_batches = create_batches(dev, dev_batch_size, max_sent_size)
        print("Number of batches:", len(train_batches))

        avg_loss_time = 0
        avg_back_time = 0
        avg_update_time = 0
        for epoch in range(num_epochs):
            print("Shuffling batches")
            random.shuffle(train_batches)
            random.shuffle(dev_batches)
            # Perform training
            print("Starting iteration", epoch + 1)
            train_words, train_loss = 0, 0.0
            start_time = time.time()
            for batch_idx, (start, length) in enumerate(train_batches):
                train_batch = train[start:start + length]

                loss_time = time.time()
                my_loss, num_words = self.model.calc_loss(train_batch)
                avg_loss_time = avg_loss_time + (time.time() - loss_time - avg_loss_time) / (batch_idx + 1)

                train_loss += my_loss.value()
                train_words += num_words

                back_time = time.time()
                my_loss.backward()
                avg_back_time = avg_back_time + (time.time() - back_time - avg_back_time) / (batch_idx + 1)

                update_time = time.time()
                self.trainer.update()
                avg_update_time = avg_update_time + (time.time() - update_time - avg_update_time) / (batch_idx + 1)

                if (batch_idx + 1) % 5 == 0:
                    print("--finished {} batches".format(batch_idx + 1))
                    print("avg_loss_time:", avg_loss_time)
                    print("avg_back_time:", avg_back_time)
                    print("avg_update_time:", avg_update_time)

            self.model.save("{}/iter_{}".format(model_path, epoch + 1))
            print("iter {}: train loss/word={:.2f}, ppl={:.4f}, time={:.2f}s".format(
                epoch + 1, train_loss / train_words, math.exp(train_loss / train_words), time.time() - start_time))
            # Evaluate on dev set
            dev_words, dev_loss = 0, 0.0
            start_time = time.time()
            for batch_idx, (start, length) in enumerate(dev_batches):
                dev_batch = dev[start:start + length]
                my_loss, num_words = self.model.calc_loss(dev_batch)
                dev_loss += my_loss.value()
                dev_words += num_words
            print("iter {}: dev loss/word={:.4f}, ppl={:.4f}, time={:.2f}s".format(
                epoch + 1, dev_loss / dev_words, math.exp(dev_loss / dev_words), time.time() - start_time))