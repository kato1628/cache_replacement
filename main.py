import torch.optim as optim
from cache_policy_model import CachePolicyModel
from configuration import config
from generator import train_data_generator

# Create training datasets generator
training_datasets = train_data_generator(config["dataset"])

# Initialize the model and optimizer
model = CachePolicyModel.from_config(config["model"])
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Train the model
for dataset in training_datasets:
    batch_size = config["training"]["batch_size"]
    
    # Chop up the dataset into batches
    subseq_length = len(dataset) // batch_size
    # (batch_size, subseq_length)
    batches = [
      dataset[i * subseq_length: (i+1) * subseq_length] for i in range(batch_size)
    ]

    for batch in batches:
        print("Training...")
        optimizer.zero_grad()
        scores, prev_reuse_distances, hidden_state = model(batch)
        print(scores)
        

        # loss = model.loss(output, cache_decision)
        # loss.backward()
        # optimizer.step()

    # print("Generating training data...")
    # for cache_state, cache_decision in dataset:
    #     print("Training...")
    #     optimizer.zero_grad()
        # output = model(cache_state.cache_access,
        #                cache_state.cache_lines,
        #                cache_state.cache_history)
        # loss = model.loss(output, cache_decision)
        # loss.backward()
        # optimizer.step()