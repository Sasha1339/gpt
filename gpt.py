import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import py7zr
import time

import keras_nlp
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 15
NUM_BATCHES = 5
EPOCHS = 5  # Can be set to a higher value for better results
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 512
MAX_GENERATION_LENGTH = 512

samsum_ds = tfds.load("samsum", split="train", data_dir="/Users/sesh/tensorflow_datasets/", download=True, as_supervised=True)

train_ds = (
    samsum_ds.map(
        lambda dialogue, summary: {"encoder_text": dialogue, "decoder_text": summary}
    )
    .batch(BATCH_SIZE)
    .cache()
)
train_ds = train_ds.take(NUM_BATCHES)

preprocessor = keras_nlp.models.BartSeq2SeqLMPreprocessor.from_preset(
    "/Users/sesh/Documents/neyro/model/bart-keras-bart_base_en-v2",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
    "/Users/sesh/Documents/neyro/model/bart-keras-bart_base_en-v2", preprocessor=preprocessor
)

bart_lm.summary()

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    epsilon=1e-6,
    global_clipnorm=1.0,  # Gradient clipping.
)
# Exclude layernorm and bias terms from weight decay.
optimizer.exclude_from_weight_decay(var_names=["bias"])
optimizer.exclude_from_weight_decay(var_names=["gamma"])
optimizer.exclude_from_weight_decay(var_names=["beta"])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bart_lm.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)

bart_lm.fit(train_ds, epochs=EPOCHS)

bart_lm.save("/Users/sesh/Documents/neyro/model/gpt/location.keras")

# bart_lm = keras.models.load_model("/Users/sesh/Documents/neyro/model/gpt/location.keras")
def generate_text(model, input_text, max_length=200, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    print(f"Total Time Elapsed: {end - start:.2f}s")
    return output



# Generate summaries.
generated_summaries = generate_text(
    bart_lm,
    "Подстанция 110/10 кВ",
    max_length=MAX_GENERATION_LENGTH,
    print_time_taken=True,
)
print(generated_summaries)