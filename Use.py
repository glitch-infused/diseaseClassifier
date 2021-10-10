from torch import nn, full, load
import pandas as pd
from typing import List


DiseaseSymptoms = pd.read_csv("./DiseaseSymptoms.csv")
SymptomSeverity = pd.read_csv("./SymptomSeverity.csv")
DiseaseDescription = pd.read_csv("./DiseaseDescription.csv")
DiseasePrecaution = pd.read_csv("./DiseasePrecaution.csv")

def GetClean(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    data = df[cols].values.flatten()
    s = pd.Series(data)
    s = s.values.reshape(df.shape)
    df = pd.DataFrame(s, columns=df.columns)
    df = df.fillna(0)
    return df

DiseaseSymptoms = GetClean(DiseaseSymptoms)
SymptomSeverity = GetClean(SymptomSeverity)

Diseases = DiseaseSymptoms["Disease"].unique()
Symptoms = SymptomSeverity["Symptom"].unique()

IN_SIZE = Symptoms.shape[0]
OUT_SIZE = Diseases.shape[0]

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

def SymptomsToInput(symptoms: List[str]):
    input = full((1, len(Symptoms)), 0.)
    for i, s in enumerate(Symptoms):
        if s in symptoms:
            input[0, i] = 1.
    return input

def SymptomsToDisease(model: Model, symptoms: List[str] = []):
    In = SymptomsToInput(symptoms)
    Out = model(In).argmax(1).item()
    return Diseases[Out]

model = Model()

with open("model.state", "rb") as f:
    model.load_state_dict(load(f))

print()
print("What are the symptoms?")
sympt = input(">>> ")
if sympt == '?':
    print("Symptoms can be:\n{}".format("\n".join(Symptoms)))
    print()
    print("What are the symptoms?")
    sympt = input(">>> ")

sympt = sympt.split(", ")
sympt = [s for s in sympt if s in Symptoms]

disease = SymptomsToDisease(model, sympt)
print()
print(f"Detected disease: {disease}")
descriptions = dict(zip(DiseaseDescription["Disease"], DiseaseDescription["Description"]))
precautions = dict(zip(
    DiseasePrecaution["Disease"],
    ((a, b, c, d) for a, b, c, d in zip(
        DiseasePrecaution["Precaution_1"],
        DiseasePrecaution["Precaution_2"], 
        DiseasePrecaution["Precaution_3"], 
        DiseasePrecaution["Precaution_4"]
    ))
))
print()
print("Description: ")
print()
print(descriptions[disease].strip('"'))
print()
print("Precautions: ", end="")
print('', *[p for p in precautions[disease] if p], sep="\n-")