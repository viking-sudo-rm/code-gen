{
  "dataset_reader": {
    "type": "char_lm",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    }
  },
  "train_data_path": "bitcoin.txt",
  # "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.dev",

  "model": {
    "type": "language_model",

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50
        },
      }
    },

    "contextualizer": {
      "type": "lstm",
      "input_size": 50,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": false
    }
  },

  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 10,
    # "patience": 10,
    "cuda_device": -1
  }
}