from models.hallucination import hallucination_guard
from data.test_dataset import test_data

correct = 0
total = len(test_data)

for item in test_data:

    # Tuple format:
    # ("question", "answer")

    question = item[0]
    expected = item[1]

    result = hallucination_guard(question)

    predicted = result["final_answer"]

    print("\nQUESTION:", question)
    print("EXPECTED:", expected)
    print("PREDICTED:", predicted)

    # Accuracy check
    if expected.lower() in predicted.lower():
        correct += 1

accuracy = (correct / total) * 100

print("\n==========================")
print("FINAL ACCURACY:", round(accuracy, 2), "%")
print("==========================")