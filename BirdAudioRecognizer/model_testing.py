from model_creation import BirdModel, BirdDataset
import torch
# Do some stuff with saved bird models
MODEL1 = BirdModel()
FILENAME = "birdmodel.pt"

# Converts the model output back into a name prediction
def predict(output, dataset):
    output = output.squeeze()
    vector = torch.zeros(dataset.unique)
    index = torch.argmax(output, dim=0).numpy()
    vector[index] = 1
    return dataset.decode(vector)

if __name__ == "__main__":
    dataset = BirdDataset('FourBirdsDataset.pkl')
    model = MODEL1
    model.load_state_dict(torch.load(FILENAME))
    model.eval()
    # Do whatever you want with the model
    count=0
    total=len(dataset)
    for i, data in enumerate(dataset):
        features, label = data
        y_pred = predict(model(features.unsqueeze(0)), dataset)
        y = dataset.decode(label.squeeze())
        if y == y_pred:
            count += 1
    print(f"Correct prediction percentage: {count/total}")