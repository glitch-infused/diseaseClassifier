print("importing libraries ...")
from typing import List
import pandas as pd
from torch import nn, full, tensor, save, rand, load
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.cuda import is_available
from torch.tensor import Tensor
from tqdm import tqdm
from random import randint

print("loading csv files ...")
#DiseaseDescription = pd.read_csv("./DiseaseDescription.csv")
#DiseasePrecaution = pd.read_csv("./DiseasePrecaution.csv")
DiseaseSymptoms = pd.read_csv("./DiseaseSymptoms.csv")
SymptomSeverity = pd.read_csv("./SymptomSeverity.csv")


print("cleaning up data ...")
def GetClean(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    data = df[cols].values.flatten()
    s = pd.Series(data)
    s = s.values.reshape(df.shape)
    df = pd.DataFrame(s, columns=df.columns)
    df = df.fillna(0)
    return df

#DiseaseDescription = GetClean(DiseaseDescription)
#DiseasePrecaution = GetClean(DiseasePrecaution)
DiseaseSymptoms = GetClean(DiseaseSymptoms)
SymptomSeverity = GetClean(SymptomSeverity)

print("preparing train data ...")
Diseases = DiseaseSymptoms["Disease"].unique()
Symptoms = SymptomSeverity["Symptom"].unique()

IN_SIZE = Symptoms.shape[0]
OUT_SIZE = Diseases.shape[0]
BATCH_SIZE = 1

def SymptomsToInput(symptoms: List[str]):
    input = full((1, len(Symptoms)), 0.)
    for i, s in enumerate(Symptoms):
        if s in symptoms:
            input[0, i] = 1.
    return input

def DiseaseToOutput(disease: str):
    return tensor(Diseases.tolist().index(disease))

class SymptomToDiseaseDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self.diseases = []
        self.symptoms = []
        for line in data.to_numpy():
            self.diseases.append(DiseaseToOutput(line[0].strip()))
            self.symptoms.append(SymptomsToInput([w.strip() for w in line[1:] if w != 0]))
        
    def __len__(self):
        return len(self.diseases)
    
    def __getitem__(self, idx):
        return self.symptoms[idx], self.diseases[idx]

DATASET = SymptomToDiseaseDataset(DiseaseSymptoms)

DATALOADER = DataLoader(DATASET, BATCH_SIZE, True)

print("building model ...")
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = round((IN_SIZE * 2 + OUT_SIZE) / 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(IN_SIZE, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            nn.Linear(hidden_size, OUT_SIZE)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        return self.softmax(self.linear(self.flatten(X)))

def SymptomsToDisease(model: Model, symptoms: List[str] = []):
    In = SymptomsToInput(symptoms)
    Out = model(In).argmax(1).item()
    return Diseases[Out]

def RandomSymptomRemove(X: Tensor):
    for item in range(X.shape[0]):
        X[item][0] = ((rand(X[item][0].shape) > 0.5) & (X[item][0] == 1.)).int()
    return X

device = "cuda" if is_available() else "cpu"

model = Model().to(device)
with open("model.state", "rb") as f:
    model.load_state_dict(load(f))
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = SGD(model.parameters(), lr=learning_rate)


print("starting training ...")
def TrainIter(model: Model, data: DataLoader):
    prog = tqdm(data)

    for X, Y in prog:
        X = RandomSymptomRemove(X)
        Logits = model(X)
        Loss = loss_fn(Logits, Y)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        prog.desc = f"loss: {Loss.item():.4f}"

    return model

def TestAccuracy(model: Model, data: DataLoader):
    prog = tqdm(data)
    score = 0
    for n_iters, (X, Y) in enumerate(prog):
        Logits = model(X)
        match = Logits.argmax(1)
        for n in match == Y:
            if n:
                score += 1
        prog.desc = f"accuracy: {score * 100 / ((n_iters + 1) * BATCH_SIZE):.4f}%"
    return score
    
score = -1
for epoch in range(100000):
    print(f"Epoch: {epoch + 1}")
    model = TrainIter(model, DATALOADER)
    TestAccuracy(model, DATALOADER)
    print()
    with open("model.state", "wb") as f:
        save(model.state_dict(), f)
