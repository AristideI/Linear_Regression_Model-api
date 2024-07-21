from datetime import date


def convertRequestToModelValue(requestDict):
    modelValues = []
    modelValues.append(1 if requestDict["diabetic"] == "yes" else 0)
    modelValues.append(requestDict["alcoholLevel"])
    modelValues.append(requestDict["heartRate"])
    modelValues.append(requestDict["bloodOxygenLevel"])
    modelValues.append(requestDict["bodyTemperature"])
    modelValues.append(requestDict["weight"])
    modelValues.append(requestDict["mriDelay"])
    modelValues.append(requestDict["age"])
    modelValues.append(convertEducation(requestDict["educationLevel"]))
    modelValues.append(1 if requestDict["dominantHand"] == "Right" else 0)
    modelValues.append(1 if requestDict["Gender"] == "Male" else 0)
    modelValues.append(0)
    modelValues.append(convertSmoking_Status(requestDict["Smoking_Status"]))
    modelValues.append(1 if requestDict["APOE_ε4"] == "Positive" else 0)
    modelValues.append(convertPhysical_Activity(requestDict["Physical_Activity"]))
    modelValues.append(1 if requestDict["Depression_Status"] == "Yes" else 0)
    modelValues.append(requestDict["Cognitive_Test_Scores"])
    modelValues.append(1 if requestDict["Medication_History"] == "Positive" else 0)
    modelValues.append(convertNutrition_Diet(requestDict["Nutrition_Diet"]))
    modelValues.append(1 if requestDict["Sleep_Quality"] == "Poor" else 0)
    modelValues.append(
        convertChronic_Health_Conditions(requestDict["Chronic_Health_Conditions"])
    )
    return modelValues


def convertEducation(value):
    if value == "Diploma/Degree":
        return 0
    elif value == "No School":
        return 1
    elif value == "Primary School":
        return 2
    elif value == "Secondary School":
        return 3
    return 1


def convertSmoking_Status(value):
    if value == "Current Smoker":
        return 0
    elif value == "Former Smoker":
        return 1
    elif value == "Never Smoked":
        return 2
    return 2


def convertPhysical_Activity(value):
    if value == "Mild Activity":
        return 0
    elif value == "Moderate Activity":
        return 1
    elif value == "Sedentary":
        return 2
    return 1


def convertNutrition_Diet(value):
    if value == "Balanced Diet":
        return 0
    elif value == "Low-Carb Diet":
        return 1
    elif value == "Mediterranean Diet":
        return 2
    return 1


def convertChronic_Health_Conditions(value):
    if value == "Diabetes":
        return 0
    elif value == "Heart Disease":
        return 1
    elif value == "Hypertension":
        return 2
    return 3


def convertPrescription(value):
    if value == 0:
        return "Donepezil"
    if value == 1:
        return "Galantamine"
    if value == 2:
        return "Memantine"
    return "Rivastigmine"


def calculate_age(birth_date: date):
    today = date.today()
    age = today.year - birth_date.year
    return age
