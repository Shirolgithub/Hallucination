from models.hallucination import hallucination_guard
from data.test_dataset import test_data

correct = 0
total = len(test_data)

for item in test_data:

    question = item["question"]
    expected = item["expected"]

    result = hallucination_guard(question)

    predicted = result["final_answer"]

    print("\nQUESTION:", question)
    print("EXPECTED:", expected)
    print("PREDICTED:", predicted)

    if expected.lower() in predicted.lower():
        correct += 1

accuracy = correct / total

print("\n==========================")
print("FINAL ACCURACY:", round(accuracy * 100, 2), "%")
print("==========================")